from typing import List, Optional

import pytorch_lightning as pl
import torch
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from torch import nn
import torch.nn.functional as F

from models.dino_vit_adapter import ViTAdapter


class FineTuner(pl.LightningModule):

    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_adapter: bool = False,
                 #adapter_config: Optional[dict] = None
                 ):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_adapter = use_adapter
        #self.adapter_config = adapter_config or {}


        if self.use_adapter:
            #self._init_adapter() 
            self.encoder = ViTAdapter()
            print(f'[ENCODER] Using encoder: ViTAdapter')
        elif dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-S14')
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-B14')
        elif dinov2_vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-L14')
        elif dinov2_vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-G14')
        else:
            raise ValueError(f'Unknown model {dinov2_vit_model}')


        # Freeze the encoder if not using adapter
        if self.use_adapter == False:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    # Not needed now to pass the init_adapter
    def _init_adapter(self):
        """Initialize ViT-Adapter with segmentation-friendly settings"""
        arch_mapping = {
            'vits14': 'vit_small',
            'vitb14': 'vit_base',
            'vitl14': 'vit_large',
            'vitg14': 'vit_giant'
        }

        vit_arch_name = arch_mapping.get(self.dinov2_vit_model)
        if vit_arch_name is None:
            raise ValueError(f'Unknown model {self.dinov2_vit_model}')

        # Default config optimized for segmentation
        default_config = {
            'pretrain_size': 518,
            'conv_inplane': 64,
            'n_points': 4,
            'deform_num_heads': 6,
            'init_values': 0.,
            'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'deform_ratio': 1.0,
            'add_vit_feature': False,
            'use_extra_extractor': True,
            'with_cp': False,
            'vit_arch_name': vit_arch_name,
            'vit_pretrained': True,
            'vit_kwargs': {
                'img_size': 518,
                'patch_size': 14,
                'init_values': 1.0,
                'ffn_layer': 'mlp',
                'block_chunks': 0,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4
            }
        }

        # Merge user config
        default_config.update(self.adapter_config)
        self.encoder = ViTAdapter(**default_config)
        self.feat_dim = self.encoder.embed_dim
        self.patch_size = self.encoder.patch_size
        self.is_adapter = True


    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])

        # Logic for ViTAdapter
        if self.use_adapter:
            f1, f2, f3, f4 = self.encoder.forward(img)
            _, _, h_f1, w_f1 = f1.shape

            # Upsample f2, f3, f4 to the same resolution as f1
            # Assuming f1, f2, f3, f4 have the same 'dim' (channels)
            f2_upsampled = F.interpolate(f2, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f3_upsampled = F.interpolate(f3, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f4_upsampled = F.interpolate(f4, size=(h_f1, w_f1), mode='bilinear', align_corners=False)

            x = torch.cat([f1, f2_upsampled, f3_upsampled, f4_upsampled], dim=1)
            return x
    
        # Default behavior for other models
        with torch.no_grad():
            block_outputs = self.encoder.forward_features(
                img,
                return_attention_features=return_attention_features,
                return_blocks=self.blocks)
            if self.blocks is None:
                block_outputs = [block_outputs]
            outs = []
            for x in block_outputs:
                x = x[feature_key]
                if feature_key == 'attn':
                    return x  # (B, num_heads, Patches+1, Patches+1)
                if feature_key in ['q', 'k', 'v']:
                    # (B, Patches+1, num_heads, feat_dim // num_heads)
                    x = x.permute((0, 2, 1, 3)).contiguous()
                    x = x.reshape((x.shape[0], -1, self.feat_dim))  # (B, Patches+1, feat_dim)
                outs.append(x)
            x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)

            x = x[:, 1:, :]  # (B, Patches, feat_dim)
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
            x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                           patches_w))  # (B, feat_dim, H, W)
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                              align_corners=False)  # (B, feat_dim, H, W)
        return x


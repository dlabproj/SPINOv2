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


from models.dino_vit_adapter import ViTAdapter


class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_adapter: bool = False,
                 adapter_config: Optional[dict] = None):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_adapter = use_adapter

        if use_adapter:
            # Use ViTAdapter instead of direct DINOv2
            if adapter_config is None:
                adapter_config = {}
            
            # Map model names to architecture names for ViTAdapter
            arch_mapping = {
                'vits14': 'vit_small',
                'vitb14': 'vit_base', 
                'vitl14': 'vit_large',
                'vitg14': 'vit_giant'
            }
            
            vit_arch_name = arch_mapping.get(dinov2_vit_model)
            if vit_arch_name is None:
                raise ValueError(f'Unknown model {dinov2_vit_model}')
            
            # Default ViTAdapter configuration
            default_config = {
                'pretrain_size': 224,
                'conv_inplane': 64,
                'n_points': 4,
                'deform_num_heads': 6,
                'init_values': 0.,
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],  # Default interaction indexes
                'with_cffn': True,
                'cffn_ratio': 0.25,
                'deform_ratio': 1.0,
                'add_vit_feature': True,
                'use_extra_extractor': True,
                'with_cp': False,
                'vit_arch_name': vit_arch_name,
                'vit_kwargs': {'patch_size': 16, 'embed_dim': self._get_embed_dim(dinov2_vit_model)},
                'vit_pretrained': True
            }
            
            # Update with user-provided config
            default_config.update(adapter_config)
            
            self.encoder = ViTAdapter(**default_config)
            self.feat_dim = self.encoder.embed_dim
            self.patch_size = self.encoder.patch_size
            
            # For adapter, we get multi-scale features
            self.is_adapter = True
            
        else:
            # Original DINOv2 implementation
            if dinov2_vit_model == 'vits14':
                self.encoder = dinov2_vits14(pretrained=True)
            elif dinov2_vit_model == 'vitb14':
                self.encoder = dinov2_vitb14(pretrained=True)
            elif dinov2_vit_model == 'vitl14':
                self.encoder = dinov2_vitl14(pretrained=True)
            elif dinov2_vit_model == 'vitg14':
                self.encoder = dinov2_vitg14(pretrained=True)
            else:
                raise ValueError(f'Unknown model {dinov2_vit_model}')

            self.feat_dim = self.encoder.num_features
            self.patch_size = self.encoder.patch_size
            self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise
            
            # Freeze backbone
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            self.is_adapter = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def _get_embed_dim(self, model_name: str) -> int:
        """Get embedding dimension for different DINOv2 models"""
        embed_dims = {
            'vits14': 384,
            'vitb14': 768,
            'vitl14': 1024,
            'vitg14': 1536
        }
        return embed_dims.get(model_name, 768)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        if self.use_adapter:
            # Use ViTAdapter forward pass
            features = self.encoder(img)  # Returns [f1, f2, f3, f4]
            
            # For semantic segmentation, we typically want the highest resolution feature
            # f1 has the highest resolution, f4 has the lowest
            # You can modify this based on your specific needs
            x = features[0]  # Use f1 (highest resolution)
            
            # If upsample_factor is specified, apply it
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, 
                                            mode='bilinear', align_corners=False)
            
            return x
        else:
            # Original DINOv2 implementation
            img_h, img_w = img.shape[2:]
            patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

            return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
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

'''
class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_adapter: bool = False):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_adapter = use_adapter

        self.adapter_config = adapter_config or {}

        if use_adapter:
            # Initialize ViT-Adapter
            vit_kwargs = {
                "img_size": 224,
                "patch_size": 14,
                "embed_dim": 768 if dinov2_vit_model in ['vits14', 'vitb14'] else 
                             1024 if dinov2_vit_model == 'vitl14' else 
                             1536 if dinov2_vit_model == 'vitg14' else 768,
                "depth": 12 if dinov2_vit_model in ['vits14', 'vitb14'] else 
                        24 if dinov2_vit_model == 'vitl14' else 
                        40 if dinov2_vit_model == 'vitg14' else 12,
                "num_heads": 12 if dinov2_vit_model in ['vits14', 'vitb14'] else 
                           16 if dinov2_vit_model == 'vitl14' else 
                           24 if dinov2_vit_model == 'vitg14' else 12,
            }
            
            self.encoder = ViTAdapter(
                vit_arch_name=f"vit_{dinov2_vit_model.replace('14', '')}",
                vit_kwargs=vit_kwargs,
                vit_pretrained=True,
                **self.adapter_config
            )
            self.feat_dim = vit_kwargs["embed_dim"]
            self.patch_size = vit_kwargs["patch_size"]

        else:
            # Original DINOv2 initialization remains the same
            if dinov2_vit_model == 'vits14':
                self.encoder = dinov2_vits14(pretrained=True)
            elif dinov2_vit_model == 'vitb14':
                self.encoder = dinov2_vitb14(pretrained=True)
            elif dinov2_vit_model == 'vitl14':
                self.encoder = dinov2_vitl14(pretrained=True)
            elif dinov2_vit_model == 'vitg14':
                self.encoder = dinov2_vitg14(pretrained=True)
            else:
                raise ValueError(f'Unknown model {dinov2_vit_model}')

            self.feat_dim = self.encoder.num_features
            self.patch_size = self.encoder.patch_size

        self.encoder.mask_token = None
        for param in self.encoder.parameters():
            param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        if self.use_adapter:
            # ViT-Adapter forward pass
            features = self.encoder(img)
            x = features[0]  # Use the finest feature map (f1)
            
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, 
                                           mode='bilinear', align_corners=False)
            return x
        else:
            # Original DINOv2 forward pass
            img_h, img_w = img.shape[2:]
            patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

            return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
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
                        return x
                    if feature_key in ['q', 'k', 'v']:
                        x = x.permute((0, 2, 1, 3)).contiguous()
                        x = x.reshape((x.shape[0], -1, self.feat_dim))
                    outs.append(x)
                x = torch.cat(outs, dim=2)
                x = x[:, 1:, :]
                x = x.permute((0, 2, 1)).contiguous()
                x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                               patches_w))
                if self.upsample_factor is not None:
                    x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, 
                                               mode='bilinear', align_corners=False)
            return x




"""

class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor

        if dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)
        elif dinov2_vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True)
        elif dinov2_vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True)
        else:
            raise ValueError(f'Unknown model {dinov2_vit_model}')

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise
        for param in self.encoder.parameters():  # freeze backbone
            param.requires_grad = True #False #Unfreezed backbone

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
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
'''

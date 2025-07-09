from typing import List, Optional

import loralib as lora
import pytorch_lightning as pl
import torch
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from torch import nn


class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, lora_enabled = True):
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

        if lora_enabled:
            self.lora_layers = nn.ModuleDict()
            apply_lora(self.encoder, self.lora_layers)
            #sets requires_grad to False for all parameters without the string "lora_" in their names
            lora.mark_only_lora_as_trainable(self.encoder)
            assert any('lora' in name.lower() for name, _ in self.named_parameters()), 'LoRA layers not found!'
            print('LoRA enabled')
        else:
            for param in self.encoder.parameters():  # freeze backbone
                param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])

        if any(param.requires_grad for param in self.encoder.parameters()):
            block_outputs = self.encoder.forward_features(
                img,
                return_attention_features=return_attention_features,
                return_blocks=self.blocks)
        else:
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


def apply_lora(model, lora_store: nn.ModuleDict, rank=4, alpha=32):
    for name, module in model.named_modules():
        if name.endswith('attn.qkv') and isinstance(module, nn.Linear):
            # name = "blocks.3.attn.qkv"
            parent_name = '.'.join(name.split('.')[:-1]) # parent_name = "blocks.3.attn"
            parent = dict(model.named_modules())[parent_name]

            lora_qkv = lora.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=rank,
                lora_alpha=alpha,
                fan_in_fan_out=False,
                bias=module.bias is not None
            )

            lora_qkv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_qkv.bias.data = module.bias.data.clone()

            unique_name = name.replace('.', '_')
            lora_store[unique_name] = lora_qkv
            setattr(parent, 'qkv', lora_store[unique_name])

# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
More info:
- https://github.com/facebookresearch/dinov2/issues/84
- https://github.com/czczup/ViT-Adapter/tree/main/segmentation
- https://github.com/czczup/ViT-Adapter/blob/ae47f7b259a87517cffce9019ee8fd492874a9cc/segmentation/mmseg_custom/models/backbones/vit_adapter.py#L20
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from external.ms_deformable_attention.modules import MSDeformAttn
from models.vit.vision_transformer import DinoVisionTransformer
from models.vit_adapter.adapter_modules import (
    InteractionBlock,
    InteractionBlockWithCls,
    SpatialPriorModule,
    deform_inputs,
)

_logger = logging.getLogger(__name__)

__all__ = ['ViTAdapter']

def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


class ViTAdapter(DinoVisionTransformer):
    def __init__(self, pretrain_size=224, conv_inplane=64, n_points=4,
                 deform_num_heads=6, init_values=0., interaction_indexes=None, with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 use_extra_extractor=True, with_cp=False,
                 vit_arch_name="vit_base", vit_kwargs=dict(), vit_pretrained=True):
        super().__init__(**vit_kwargs)

        model_name = _make_dinov2_model_name(vit_arch_name, vit_kwargs["patch_size"])
        if vit_pretrained:
            url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            self.load_state_dict(state_dict, strict=True)
            """
            # Handle positional embedding size mismatch
            if state_dict['pos_embed'].shape[1] != self.pos_embed.shape[1]:
                orig_size = int((state_dict['pos_embed'].shape[1] - 1) ** 0.5)
                new_size = int((self.pos_embed.shape[1] - 1) ** 0.5)
                
                # Class token stays the same
                pos_embed = state_dict['pos_embed'][:, 0:1]
                
                # Interpolate patch tokens
                patch_embed = state_dict['pos_embed'][:, 1:]
                patch_embed = patch_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                patch_embed = F.interpolate(
                    patch_embed, 
                    size=(new_size, new_size), 
                    mode='bicubic', 
                    align_corners=False
                )
                patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_size*new_size, -1)
                
                # Combine class token and interpolated patch tokens
                state_dict['pos_embed'] = torch.cat([pos_embed, patch_embed], dim=1)
        
            # Handle block naming convention differences
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('blocks.'):
                    parts = k.split('.')
                    if parts[1].isdigit() and not parts[2].isdigit():  # blocks.0.attn case
                        new_key = f"{parts[0]}.{parts[1]}.0.{'.'.join(parts[2:])}"
                        new_state_dict[new_key] = v
                    else:
                        new_state_dict[k] = v
                else:
                    new_state_dict[k] = v
            
            # Filter out LayerScale parameters
            new_state_dict = {
                k: v for k, v in new_state_dict.items() 
                if not k.endswith('ls1.gamma') and not k.endswith('ls2.gamma')
            }
            
            # Load with strict=False to ignore remaining mismatches
            msg = self.load_state_dict(new_state_dict, strict=False)
            _logger.info(f"Loaded pretrained weights with message: {msg}")
        """

        # Freeze vit
        for param in self.parameters():
            param.requires_grad = False

        # self.num_classes = 80
        self.mask_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def init_weights(self) -> None:
        pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 14, self.pretrain_size[1] // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        # Ensure input is 4D [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension if missing

        # Get sp atial dimensions
        bs, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0

        #deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        _, _, h, w = x.shape
        x = self.patch_embed(x)
        W_vit = w // self.patch_size
        H_vit = h // self.patch_size
        W_adapter = w // 16
        H_adapter = h // 16
        bs, n, dim = x.shape

    # Generate deform inputs with correct spatial shapes
        spatial_shapes = [
            [H_adapter*2, W_adapter*2],  # For c2
            [H_adapter, W_adapter],      # For c3
            [H_adapter//2, W_adapter//2] # For c4
        ]

        # Calculate the actual spatial dimensions from the patch embeddings
        # This is the key fix - ensure we're using the correct dimensions

        # Calculate patch shape from sequence length
        h_patch = int(math.sqrt(n))
        w_patch = int(math.sqrt(n))
        if h_patch * w_patch != n:
            # If not a perfect square, find the closest factors
            factors = []
            for i in range(1, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    factors.append((i, n // i))
            if factors:
                h_patch, w_patch = factors[-1]  # Take the most square-like
            else:
                h_patch, w_patch = n, 1

        # For patch tokens, use their shape
        patch_spatial_shapes = [[h_patch, w_patch]]

        # For multi-scale features, use multi-scale shapes
        multi_scale_shapes = [
            [H_adapter*2, W_adapter*2],  # For c2
            [H_adapter, W_adapter],      # For c3
            [H_adapter//2, W_adapter//2] # For c4
        ]

        deform_inputs1, deform_inputs2 = deform_inputs(x, spatial_shapes=spatial_shapes)


        #pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        pos_embed = self.interpolate_pos_encoding(x, w, h)  # Use parent class method
        pos_embed = pos_embed[:, 1:]  # Remove cls token pos embed

        x = x + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapter, W_adapter)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_adapter * 2, W_adapter * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_adapter // 2, W_adapter // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]


class Identity(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

    def forward(self, x):
        return x

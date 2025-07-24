import os

import torch
from segment_anything import sam_model_registry

DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "sam_checkpoints")

_SAM_CHECKPOINTS = {
    "vit_h": os.path.join(DEFAULT_CHECKPOINT_DIR, "sam_vit_h_4b8939.pth"),
    "vit_l": os.path.join(DEFAULT_CHECKPOINT_DIR, "sam_vit_l_0b3195.pth"),
    "vit_b": os.path.join(DEFAULT_CHECKPOINT_DIR, "sam_vit_b_01ec64.pth"),
}


def _make_sam_model(
        model_type: str = "vit_h",
        checkpoint_path: str = None,
        pretrained: bool = True,
):
    """
    Load SAM encoder (image encoder only) from pretrained checkpoint.
    """
    if checkpoint_path is None:
        if model_type not in _SAM_CHECKPOINTS:
            raise ValueError(f"Unknown SAM model_type: {model_type}")
        checkpoint_path = _SAM_CHECKPOINTS[model_type]
        # print(f"[DEBUG] Loaded SAM weights from: {checkpoint_path}")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    # print("SAM patch embed weight (mean):", sam.image_encoder.patch_embed.proj.weight.mean().item())
    image_encoder = sam.image_encoder

    return image_encoder


def sam_vit_h(*, pretrained: bool = True, checkpoint_path: str = None):
    """
    SAM ViT-H image encoder.
    """
    return _make_sam_model(model_type="vit_h", checkpoint_path=checkpoint_path, pretrained=pretrained)


def sam_vit_l(*, pretrained: bool = True, checkpoint_path: str = None):
    """
    SAM ViT-L image encoder.
    """
    return _make_sam_model(model_type="vit_l", checkpoint_path=checkpoint_path, pretrained=pretrained)


def sam_vit_b(*, pretrained: bool = True, checkpoint_path: str = None):
    """
    SAM ViT-B image encoder.
    """
    return _make_sam_model(model_type="vit_b", checkpoint_path=checkpoint_path, pretrained=pretrained)

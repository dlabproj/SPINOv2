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
        model_type: str,
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

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    # print("SAM patch embed weight (mean):", sam.image_encoder.patch_embed.proj.weight.mean().item())
    print("Pixel Mean:", sam.pixel_mean.view(-1).tolist())
    print("Pixel Std:", sam.pixel_std.view(-1).tolist())

    # 2. Patch image size for your 448x896 inputs
    new_h, new_w = 448, 896
    patch_size = sam.image_encoder.patch_embed.proj.kernel_size[0]  # 16
    new_embedding_h = new_h // patch_size
    new_embedding_w = new_w // patch_size

    # 3. Update image size expectations inside SAM and its modules
    sam.image_encoder.img_size = new_h  # Note: Sam uses square img_size but accepts rectangular input
    sam.prompt_encoder.input_image_size = (new_h, new_w)
    sam.prompt_encoder.image_embedding_size = (new_embedding_h, new_embedding_w)

    return sam.image_encoder


def sam_vit_b(*, pretrained: bool = True, checkpoint_path: str = None):
    """
    SAM ViT-B image encoder.
    """
    return _make_sam_model(model_type="vit_b", checkpoint_path=checkpoint_path, pretrained=pretrained)


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


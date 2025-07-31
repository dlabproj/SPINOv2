import torch
import timm
from timm.data import resolve_data_config


def _make_eva_model(
        model_name: str,
        pretrained: bool = True,
        **kwargs,
):
    model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
    # config = resolve_data_config({}, model=model)
    # print(config["mean"], config["std"])
    return model


def eva02_vitb14(pretrained: bool = True, **kwargs):
    """
    EVA-02 ViT-B/14 model.
    """
    return _make_eva_model(model_name="eva02_base_patch14_448.mim_in22k_ft_in22k", pretrained=pretrained, **kwargs)


def eva02_vitl14(pretrained: bool = True, **kwargs):
    """
    EVA-02 ViT-L/14 model.
    """
    return _make_eva_model(model_name="eva02_large_patch14_448.mim_in22k_ft_in22k", pretrained=pretrained, **kwargs)

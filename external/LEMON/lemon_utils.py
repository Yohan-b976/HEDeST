from __future__ import annotations

import json
from pathlib import Path

import moco_vits
import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from torch import nn
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import CenterCrop
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import Normalize


def _clean_moco_state_dict(state_dict: dict[str, Tensor], linear_keyword: str) -> dict[str, Tensor]:
    """
    Filters and renames keys from a MoCo state_dict.

    It selects keys from the 'base_encoder', removes the given linear layer keyword,
    and strips the 'module.base_encoder.' prefix.
    """
    for key in list(state_dict.keys()):
        # Check if the key belongs to the base encoder's backbone
        if key.startswith("module.base_encoder") and not key.startswith(f"module.base_encoder.{linear_keyword}"):
            # Create a new key by stripping the prefix
            new_key = key[len("module.base_encoder.") :]
            state_dict[new_key] = state_dict[key]

        # Delete the old key (either renamed or unused)
        del state_dict[key]

    return state_dict


def get_params_group_for_adamw(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bias = n.endswith(".bias")
        is_norm = ("norm" in n.lower()) or ("bn" in n.lower())
        is_vit_special = ("pos_embed" in n) or ("cls_token" in n)
        if p.ndim <= 1 or is_bias or is_norm or is_vit_special:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def load_moco_encoder(
    model: nn.Module,
    weight_path: Path,
    linear_keyword: str,
) -> nn.Module:
    """
    Loads pre-trained MoCo weights into a given model instance (ResNet, ViT, etc.).

    This function handles loading the checkpoint, cleaning the state dictionary keys,
    and loading the weights into the model's backbone. It finishes by replacing
    the model's linear head with an Identity layer to turn it into a feature extractor.

    Args:
        model: An instantiated PyTorch model (e.g., from timm or a custom module).
        weight_path: Path to the .pth or .pt MoCo checkpoint file.
        linear_keyword: The name of the final linear layer to exclude (e.g., 'fc' or 'head').

    Returns:
        The same model, with pre-trained backbone weights and the head replaced
        by nn.Identity(), ready for feature extraction.
    """
    if not weight_path.exists():
        logger.info(f"No checkpoint found at {weight_path}, downloading from hugging face...")
        get_lemon_checkpoint_path(weight_path.parent)

    logger.info(f"=> Loading MoCo checkpoint from '{weight_path}'")

    # Use weights_only=True for added security if the checkpoint doesn't contain pickled code
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)

    # Extract the state dictionary containing the model weights
    state_dict = checkpoint["state_dict"]

    # Clean the state_dict to match the model's architecture
    cleaned_state_dict = _clean_moco_state_dict(state_dict, linear_keyword)

    # Load the cleaned weights into the model
    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    logger.info(msg)
    logger.info("=> Successfully loaded pre-trained model backbone.")

    # Replace the model's head to turn it into a feature extractor
    if hasattr(model, linear_keyword):
        setattr(model, linear_keyword, nn.Identity())
        logger.info(f"=> Model's '{linear_keyword}' layer replaced with nn.Identity for feature extraction.")

    return model


def get_vit_feature_extractor(weight_path: Path, model_name: str, img_size: int) -> nn.Module:
    """Creates a ViT feature extractor using the unified loader."""
    # 1. Create the model architecture shell
    vit_model = moco_vits.__dict__[model_name](img_size=img_size, num_classes=0)

    # 2. Use the unified function to load weights and prepare for feature extraction
    feature_extractor = load_moco_encoder(model=vit_model, weight_path=weight_path, linear_keyword="head")
    return feature_extractor


def get_lemon_checkpoint_path(output_dir: Path) -> None:
    logger.info(f"Downloading lemon/lemon.pth.tar to {output_dir} ...")
    tmp_path = hf_hub_download(
        repo_id="iclr2025-anonymous/LEMON",
        filename="lemon.pth.tar",
    )
    # Copy from cache to your desired location
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lemon.pth.tar"
    output_path.write_bytes(Path(tmp_path).read_bytes())


def prepare_transform(
    stats_path,
    size: int = 40,
) -> Compose:
    # Get normalisation stats
    with open(stats_path, "r") as f:
        norm_dict = json.load(f)
    mean = norm_dict["mean"]
    std = norm_dict["std"]

    # Prepare transform
    list_transform = [
        ToTensor(),
        Normalize(mean=mean, std=std),
        CenterCrop(size=size),
    ]
    transform = Compose(list_transform)
    return transform

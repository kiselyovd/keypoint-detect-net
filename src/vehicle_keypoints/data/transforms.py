"""Image transforms for training and inference."""
from __future__ import annotations

import torch
from torchvision.transforms import v2


def build_train_transforms(image_size: int = 224) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(image_size, antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transforms(image_size: int = 224) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(int(image_size * 1.14), antialias=True),
        v2.CenterCrop(image_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 128
    val_ratio: float = 0.1
    num_workers: int = 2
    seed: int = 42


def get_fashion_mnist_datasets() -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    # For attacks, prefer *no normalization* to keep epsilon interpretation simple.
    transform = transforms.ToTensor()

    train_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_ds, test_ds


def split_train_val(
    train_ds: torch.utils.data.Dataset,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset]]:
    if val_ratio <= 0.0:
        return train_ds, None

    g = torch.Generator().manual_seed(seed)
    val_size = int(len(train_ds) * val_ratio)
    train_size = len(train_ds) - val_size
    train_subset, val_subset = random_split(train_ds, [train_size, val_size], generator=g)
    return train_subset, val_subset


def make_loaders(
    train_subset: torch.utils.data.Dataset,
    val_subset: Optional[torch.utils.data.Dataset],
    test_ds: torch.utils.data.Dataset,
    cfg: DataConfig,
    device: torch.device,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    pin = (device.type == "cuda")

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = None
    if val_subset is not None and len(val_subset) > 0:
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            persistent_workers=(cfg.num_workers > 0),
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, val_loader, test_loader

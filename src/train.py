from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    optimizer: str = "adam"  # "adam" or "sgd"


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> optim.Optimizer:
    name = cfg.optimizer.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr)
    raise ValueError("optimizer must be 'adam' or 'sgd'")


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

    return running_loss / max(1, len(loader))


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    cfg: TrainConfig,
) -> Dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)

    history = {"train_loss": [], "val_acc": []}

    from .eval import accuracy  # local import to avoid circular deps

    best_val = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history["train_loss"].append(loss)

        if val_loader is not None:
            val = accuracy(model, val_loader, device)
            history["val_acc"].append(val)

            if val > best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best model (if val exists)
    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_acc": best_val}

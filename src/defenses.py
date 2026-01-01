from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from .attacks import fgsm, AttackConfig


@torch.no_grad()
def add_gaussian_noise(x: torch.Tensor, sigma: float = 0.05, clamp_min: float = 0.0, clamp_max: float = 1.0):
    return (x + sigma * torch.randn_like(x)).clamp(clamp_min, clamp_max)


@dataclass(frozen=True)
class AdvTrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    optimizer: str = "adam"
    attack_eps: float = 0.1


def _make_optimizer(model: torch.nn.Module, cfg: AdvTrainConfig) -> optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr)
    raise ValueError("optimizer must be 'adam' or 'sgd'")


def adversarial_train_fgsm(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    cfg: AdvTrainConfig,
) -> Dict[str, Any]:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg)

    from .eval import accuracy

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    atk_cfg = AttackConfig(eps=cfg.attack_eps)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # generate adversarial examples on-the-fly
            x_adv = fgsm(model, x, y, atk_cfg)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        history["train_loss"].append(running / max(1, len(train_loader)))

        if val_loader is not None:
            val = accuracy(model, val_loader, device)
            history["val_acc"].append(val)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_acc": best_val}


def adversarial_train_pgd(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    cfg: AdvTrainConfig,
) -> Dict[str, Any]:
    """
    Adversarial training using PGD attack for stronger robustness.
    
    PGD-based adversarial training typically provides better robustness
    than FGSM-based training, at the cost of longer training time.
    """
    from .attacks import pgd_linf, AttackConfig as AtkCfg
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg)

    from .eval import accuracy

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    # PGD config: more steps for stronger attack
    atk_cfg = AtkCfg(eps=cfg.attack_eps, steps=7, step_size=cfg.attack_eps/4)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # generate adversarial examples using PGD
            x_adv = pgd_linf(model, x, y, atk_cfg)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_adv)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        history["train_loss"].append(running / max(1, len(train_loader)))

        if val_loader is not None:
            val = accuracy(model, val_loader, device)
            history["val_acc"].append(val)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_acc": best_val}


def mixed_adversarial_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    cfg: AdvTrainConfig,
    mix_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    Mixed adversarial training: train on both clean and adversarial examples.
    
    Args:
        mix_ratio: fraction of batch to use as adversarial (0.5 = 50% clean, 50% adv)
    
    This approach often provides better clean accuracy while maintaining robustness.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg)

    from .eval import accuracy

    history = {"train_loss": [], "val_acc": []}
    best_val = -1.0
    best_state = None

    atk_cfg = AttackConfig(eps=cfg.attack_eps)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Split batch
            split = int(x.size(0) * mix_ratio)
            x_clean, y_clean = x[split:], y[split:]
            x_adv_src, y_adv = x[:split], y[:split]
            
            # Generate adversarial examples for portion of batch
            if split > 0:
                x_adv = fgsm(model, x_adv_src, y_adv, atk_cfg)
                x_combined = torch.cat([x_adv, x_clean], dim=0)
                y_combined = torch.cat([y_adv, y_clean], dim=0)
            else:
                x_combined, y_combined = x_clean, y_clean

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_combined)
            loss = criterion(logits, y_combined)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        history["train_loss"].append(running / max(1, len(train_loader)))

        if val_loader is not None:
            val = accuracy(model, val_loader, device)
            history["val_acc"].append(val)
            if val > best_val:
                best_val = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"history": history, "best_val_acc": best_val}


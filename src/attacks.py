from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AttackConfig:
    eps: float = 0.1
    steps: int = 10       # PGD steps
    step_size: float = 0.01
    clamp_min: float = 0.0
    clamp_max: float = 1.0


def fgsm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AttackConfig,
) -> torch.Tensor:
    model.eval()
    x_adv = x.detach().clone().requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    loss.backward()

    with torch.no_grad():
        x_adv = x_adv + cfg.eps * x_adv.grad.sign()
        x_adv = x_adv.clamp(cfg.clamp_min, cfg.clamp_max)

    return x_adv.detach()


def pgd_linf(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: AttackConfig,
) -> torch.Tensor:
    model.eval()

    x_orig = x.detach()
    x_adv = x_orig.clone()

    # random start helps make PGD stronger
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-cfg.eps, cfg.eps)
    x_adv = x_adv.clamp(cfg.clamp_min, cfg.clamp_max)

    for _ in range(cfg.steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + cfg.step_size * grad.sign()
            # project back into eps-ball around x_orig
            delta = torch.clamp(x_adv - x_orig, min=-cfg.eps, max=cfg.eps)
            x_adv = (x_orig + delta).clamp(cfg.clamp_min, cfg.clamp_max)

    return x_adv.detach()

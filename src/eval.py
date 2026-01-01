from __future__ import annotations
from typing import Dict, Any, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return 100.0 * correct / max(1, total)


@torch.no_grad()
def auc_class_k_vs_rest(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    k: int = 2,
) -> float:
    model.eval()
    y_true, y_score = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, k]
        y_score.append(probs.detach().cpu().numpy())
        y_true.append((y == k).numpy().astype(int))

    y_true = np.concatenate(y_true).astype(np.int32)
    y_score = np.concatenate(y_score).astype(np.float32)

    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


@torch.no_grad()
def robust_accuracy_from_adv(
    model: torch.nn.Module,
    adv_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    # adv_loader should yield (x_adv, y)
    return accuracy(model, adv_loader, device)


@torch.no_grad()
def confidence_stats(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute confidence statistics (mean and std of max softmax probability).
    
    Useful for AI safety analysis: overconfident wrong predictions are dangerous.
    """
    model.eval()
    max_probs = []
    
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1)
        max_probs.append(max_prob.cpu())
    
    all_probs = torch.cat(max_probs)
    return {
        "mean_confidence": float(all_probs.mean().item()),
        "std_confidence": float(all_probs.std().item()),
        "min_confidence": float(all_probs.min().item()),
        "max_confidence": float(all_probs.max().item()),
    }


def attack_success_rate(
    model: torch.nn.Module,
    clean_loader: torch.utils.data.DataLoader,
    attack_fn,
    attack_cfg,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Compute attack success rate: % of correctly classified samples that become misclassified.
    
    Returns success rate and counts for detailed analysis.
    """
    model.eval()
    originally_correct = 0
    fooled = 0
    
    for x, y in clean_loader:
        x, y = x.to(device), y.to(device)
        
        # Check original predictions
        with torch.no_grad():
            clean_logits = model(x)
            clean_preds = clean_logits.argmax(dim=1)
            correct_mask = (clean_preds == y)
            originally_correct += int(correct_mask.sum().item())
        
        # Generate adversarial examples
        x_adv = attack_fn(model, x, y, attack_cfg)
        
        # Check adversarial predictions
        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_preds = adv_logits.argmax(dim=1)
            # Fooled = was correct, now wrong
            fooled += int((correct_mask & (adv_preds != y)).sum().item())
    
    rate = 100.0 * fooled / max(1, originally_correct)
    return {
        "attack_success_rate": rate,
        "originally_correct": originally_correct,
        "fooled": fooled,
    }


@torch.no_grad()
def per_class_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """
    Compute per-class accuracy to identify failure modes.
    
    Returns dict with per-class accuracy and confusion info.
    """
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        for c in range(num_classes):
            mask = (y == c)
            total[c] += mask.sum().item()
            correct[c] += ((preds == y) & mask).sum().item()
    
    per_class = (100.0 * correct / total.clamp(min=1)).tolist()
    return {
        "per_class_accuracy": per_class,
        "worst_class": int(torch.argmin(correct / total.clamp(min=1)).item()),
        "best_class": int(torch.argmax(correct / total.clamp(min=1)).item()),
    }


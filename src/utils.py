from __future__ import annotations
import random
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt

# FashionMNIST class names for visualization
FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # reproducibility knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_figure(fig: plt.Figure, name: str, reports_dir: str = "reports/figures") -> str:
    """Save figure to reports/figures directory. Returns the saved path."""
    path = Path(reports_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{name}.png"
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
    return str(filepath)


def plot_adversarial_examples(
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    y_true: torch.Tensor,
    preds_clean: torch.Tensor,
    preds_adv: torch.Tensor,
    num_samples: int = 5,
    eps: float = 0.0,
) -> plt.Figure:
    """
    Plot clean vs adversarial examples side-by-side.
    Shows original image, perturbation (amplified), and adversarial result.
    """
    fig, axes = plt.subplots(3, num_samples, figsize=(2.5 * num_samples, 7))
    
    for i in range(min(num_samples, x_clean.shape[0])):
        clean = x_clean[i].cpu().squeeze().numpy()
        adv = x_adv[i].cpu().squeeze().numpy()
        delta = (adv - clean)
        
        # Original
        axes[0, i].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Clean\n{FASHION_MNIST_CLASSES[y_true[i]]}", fontsize=9)
        axes[0, i].axis("off")
        
        # Perturbation (amplified for visibility)
        axes[1, i].imshow(delta, cmap="RdBu", vmin=-0.3, vmax=0.3)
        axes[1, i].set_title(f"Perturbation\n(ε={eps:.3f})", fontsize=9)
        axes[1, i].axis("off")
        
        # Adversarial
        pred_label = FASHION_MNIST_CLASSES[preds_adv[i]]
        color = "green" if preds_adv[i] == y_true[i] else "red"
        axes[2, i].imshow(adv, cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title(f"Adversarial\nPred: {pred_label}", fontsize=9, color=color)
        axes[2, i].axis("off")
    
    fig.suptitle("Adversarial Examples: Clean → Perturbation → Result", fontsize=12)
    plt.tight_layout()
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot training loss and validation accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curve
    ax1.plot(epochs, history["train_loss"], "b-", marker="o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    if history.get("val_acc"):
        ax2.plot(epochs, history["val_acc"], "g-", marker="o", markersize=4)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_title("Validation Accuracy")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No validation data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Validation Accuracy")
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_epsilon_vs_accuracy(
    epsilons: List[float],
    clean_acc: float,
    fgsm_accs: List[float],
    pgd_accs: Optional[List[float]] = None,
) -> plt.Figure:
    """Plot accuracy vs epsilon for different attacks."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.axhline(y=clean_acc, color="gray", linestyle="--", label=f"Clean ({clean_acc:.1f}%)")
    ax.plot(epsilons, fgsm_accs, "b-o", label="FGSM", markersize=6)
    
    if pgd_accs is not None:
        ax.plot(epsilons, pgd_accs, "r-s", label="PGD", markersize=6)
    
    ax.set_xlabel("Perturbation (ε)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Robustness: Accuracy vs. Perturbation Strength", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    return fig


def plot_robustness_comparison(
    model_names: List[str],
    clean_accs: List[float],
    robust_accs: List[float],
    attack_name: str = "FGSM (ε=0.1)",
) -> plt.Figure:
    """Bar chart comparing clean vs robust accuracy across models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, clean_accs, width, label="Clean Accuracy", color="steelblue")
    bars2 = ax.bar(x + width/2, robust_accs, width, label=f"Robust ({attack_name})", color="coral")
    
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Clean vs. Robust Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="best")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


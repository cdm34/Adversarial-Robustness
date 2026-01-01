"""
Adversarial Robustness Research - Core Module

Provides models, attacks, defenses, training, and evaluation utilities
for studying adversarial robustness on FashionMNIST.
"""
from __future__ import annotations

# --- Models ---
from .models import FashionMNISTNet, FashionMNISTNetDropout

# --- Data ---
from .data import DataConfig, get_fashion_mnist_datasets, split_train_val, make_loaders

# --- Attacks ---
from .attacks import AttackConfig, fgsm, pgd_linf

# --- Training ---
from .train import TrainConfig, fit, train_one_epoch

# --- Evaluation ---
from .eval import (
    accuracy, 
    auc_class_k_vs_rest, 
    robust_accuracy_from_adv,
    confidence_stats,
    attack_success_rate,
    per_class_accuracy,
)

# --- Defenses ---
from .defenses import (
    AdvTrainConfig, 
    adversarial_train_fgsm, 
    adversarial_train_pgd,
    mixed_adversarial_training,
    add_gaussian_noise,
)

# --- Utilities ---
from .utils import (
    get_device, 
    set_seed,
    save_figure,
    plot_adversarial_examples,
    plot_training_curves,
    plot_epsilon_vs_accuracy,
    plot_robustness_comparison,
    FASHION_MNIST_CLASSES,
)

__all__ = [
    # Models
    "FashionMNISTNet",
    "FashionMNISTNetDropout",
    # Data
    "DataConfig",
    "get_fashion_mnist_datasets",
    "split_train_val",
    "make_loaders",
    # Attacks
    "AttackConfig",
    "fgsm",
    "pgd_linf",
    # Training
    "TrainConfig",
    "fit",
    "train_one_epoch",
    # Evaluation
    "accuracy",
    "auc_class_k_vs_rest",
    "robust_accuracy_from_adv",
    "confidence_stats",
    "attack_success_rate",
    "per_class_accuracy",
    # Defenses
    "AdvTrainConfig",
    "adversarial_train_fgsm",
    "adversarial_train_pgd",
    "mixed_adversarial_training",
    "add_gaussian_noise",
    # Utilities
    "get_device",
    "set_seed",
    "save_figure",
    "plot_adversarial_examples",
    "plot_training_curves",
    "plot_epsilon_vs_accuracy",
    "plot_robustness_comparison",
    "FASHION_MNIST_CLASSES",
]

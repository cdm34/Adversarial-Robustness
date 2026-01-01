# Adversarial Robustness Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/01_baseline_cnn.ipynb)

An AI safety research project studying adversarial robustness in deep learning systems using FashionMNIST.

## Research Questions

1. **How does adversarial noise affect CNN accuracy and confidence?**
2. **How do different defenses (dropout, preprocessing, adversarial training) change robustness?**
3. **What trade-offs exist between clean accuracy and adversarial robustness?**

## Project Structure

```
adversarial-robustness/
├── notebooks/
│   ├── 01_baseline_cnn.ipynb         # Train baseline CNN
│   ├── 02_fgsm_pgd_attacks.ipynb     # Attack evaluation
│   ├── 03_defenses_dropout-preproc.ipynb  # Defense strategies
│   └── 04_adversarial_training.ipynb # Adversarial training
├── src/
│   ├── models.py      # CNN architectures
│   ├── attacks.py     # FGSM / PGD implementations
│   ├── defenses.py    # Adversarial training & preprocessing
│   ├── train.py       # Training loops
│   ├── eval.py        # Accuracy, robustness, confidence metrics
│   ├── data.py        # Dataset loading
│   └── utils.py       # Visualization & helpers
├── reports/figures/   # Saved visualizations
└── checkpoints/       # Saved models
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_baseline_cnn.ipynb
```

## Key Features

- **Modular `src/` library**: Reusable code for models, attacks, defenses, and evaluation
- **Research-oriented notebooks**: Focus on experiments and interpretation
- **AI safety metrics**: Confidence analysis, attack success rates, per-class accuracy
- **Reproducible**: Seeds and checkpoints for consistent results

## Attacks Implemented

- **FGSM** (Fast Gradient Sign Method)
- **PGD** (Projected Gradient Descent with random start)

## Defenses Implemented

- Dropout regularization
- Gaussian noise preprocessing
- FGSM adversarial training
- PGD adversarial training
- Mixed adversarial training (clean + adversarial)

## Technical Stack

- PyTorch
- FashionMNIST dataset
- Google Colab / Jupyter compatible

# Running This Project in Google Colab

## Quick Start (Recommended)

Click the badge below to open the first notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/01_baseline_cnn.ipynb)

---

## Option 1: Open Notebooks Directly from GitHub

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Open notebook → GitHub**
3. Enter: `cdm34/adversarial-robustness`
4. Select the notebook you want to run

### Direct Links to All Notebooks:
- [01_baseline_cnn.ipynb](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/01_baseline_cnn.ipynb)
- [02_fgsm_pgd_attacks.ipynb](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/02_fgsm_pgd_attacks.ipynb)
- [03_defenses_dropout-preproc.ipynb](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/03_defenses_dropout-preproc.ipynb)
- [04_adversarial_training.ipynb](https://colab.research.google.com/github/cdm34/adversarial-robustness/blob/main/notebooks/04_adversarial_training.ipynb)

---

## Option 2: Clone the Full Repository

Add this cell at the top of your Colab notebook:

```python
# Clone the repository and install dependencies
!git clone https://github.com/cdm34/adversarial-robustness.git
%cd adversarial-robustness
!pip install -q -r requirements.txt

# Make src/ importable
import sys
sys.path.insert(0, '/content/adversarial-robustness')
```

---

## Enable GPU Acceleration

For faster training, enable GPU:

1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

Verify GPU is working:
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

---

## Recommended Order

Run the notebooks in this order:

1. **01_baseline_cnn** – Train the baseline CNN model
2. **02_fgsm_pgd_attacks** – Evaluate adversarial attacks
3. **03_defenses_dropout-preproc** – Test defense strategies
4. **04_adversarial_training** – Adversarial training experiments

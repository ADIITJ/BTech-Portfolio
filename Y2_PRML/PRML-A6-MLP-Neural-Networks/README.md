# PRML — Assignment 6: Multilayer Perceptron & Neural Networks

**Subject:** Pattern Recognition & Machine Learning (Y2)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Topics Covered
- Multilayer Perceptron (MLP) — with and without activation functions
- Backpropagation, SGD / Adam optimizer
- Effect of ReLU activations on convergence and accuracy

## Problem — MNIST Digit Classification (PyTorch)

Two MLP architectures trained on MNIST (60k digits, 10 classes):

| Model | Architecture | Activations |
|---|---|---|
| MLP1 | 784 → 1024 → 512 → 10 | None (linear only) |
| MLP2 | 784 → 1024 → 512 → 10 | ReLU between layers |

**Training setup:**
- 80/20 train/validation split (48k / 12k)
- Test set: 10k
- Data augmentation: random rotation (±10°), random crop with padding
- Optimizer: Adam · Loss: Cross-Entropy
- Tracked: training loss, validation loss, accuracy per epoch

## Key Findings
ReLU activations significantly improve convergence speed and final accuracy by breaking the linearity of stacked linear layers.

## Tech Stack
- Python, PyTorch, torchvision, Matplotlib, NumPy

## Key Files
- `B22AI045_all.ipynb` — full training + comparison notebook
- `b22ai045_prob1.py` — standalone script
- `B22AI045_Report.pdf` — assignment report
- `alt-submission/` — alternate version of the same assignment

## Run
```bash
pip install torch torchvision matplotlib
jupyter notebook B22AI045_all.ipynb
```

# PRML — Assignment 3: Perceptron & PCA-Based Face Recognition

**Subject:** Pattern Recognition & Machine Learning (Y2)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Topics Covered
- Perceptron learning algorithm (binary classifier from scratch)
- Principal Component Analysis (PCA) for dimensionality reduction
- K-Nearest Neighbors (KNN) classification

## Problems

### Q1 — Perceptron Algorithm
Custom Perceptron implementation for binary classification:
- Z-score normalization
- Iterative weight update with convergence tracking
- Boundary visualization

### Q2 — PCA + KNN for Face Recognition
Dimensionality reduction on the **LFW (Labeled Faces in the Wild)** dataset:
- PCA with varying components (150–950) to find ≥95% explained variance
- KNN classification on reduced embeddings
- Accuracy vs. component count trade-off analysis

## Tech Stack
- Python, NumPy, scikit-learn, Matplotlib

## Key Files
- `B22AI045_problem2.py` — PCA + KNN face recognition
- `B22AI045_test.py` — test harness
- `B22AI045_README.txt` — original notes
- `B22AI045_report.pdf` — assignment report

## Run
```bash
pip install scikit-learn matplotlib numpy
python B22AI045_problem2.py
```

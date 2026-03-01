# PRML — Assignment 4: Linear Discriminant Analysis & Naive Bayes

**Subject:** Pattern Recognition & Machine Learning (Y2)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Topics Covered
- Linear Discriminant Analysis (LDA) — from scratch
- Gaussian Naive Bayes (custom + sklearn comparison)
- Multinomial Naive Bayes with Laplace smoothing

## Problems

### Q1 — LDA from Scratch on Iris Dataset
Custom LDA implementation computing:
- Class-wise mean vectors
- Within-class scatter matrix **S_w**
- Between-class scatter matrix **S_b**
- Eigenvectors of S_w⁻¹·S_b for projection into discriminant space
- 2D projection visualization vs. PCA comparison

### Q2 — Naive Bayes Classifiers
- **Gaussian NB (custom):** class priors + per-feature Gaussian likelihoods
- **Multinomial NB (custom):** with and without Laplace smoothing
- Benchmarked against `sklearn.naive_bayes.GaussianNB` and `MultinomialNB`
- KNN classification added as baseline comparison

## Tech Stack
- Python, NumPy, scikit-learn, Matplotlib

## Key Files
- `B22AI045_all.ipynb` — full implementation
- `b22ai045_myLDA.py` — standalone LDA implementation
- `B22AI045_report.pdf` — assignment report

## Run
```bash
jupyter notebook B22AI045_all.ipynb
# or standalone LDA:
python b22ai045_myLDA.py
```

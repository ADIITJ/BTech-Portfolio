# PRML — Assignment 5: K-Means Clustering & Support Vector Machines

**Subject:** Pattern Recognition & Machine Learning (Y2)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Topics Covered
- K-Means Clustering (from scratch) with image compression application
- Support Vector Machines — Linear, Polynomial, RBF kernels
- Hyperparameter tuning with RandomizedSearchCV

## Problems

### Q1 — K-Means for Image Compression
- Custom K-Means from scratch (color quantization)
- Compresses images by replacing pixel colors with cluster centroids
- Gaussian filter preprocessing
- Benchmarked against `sklearn.cluster.KMeans` on execution time

### Q2 — SVM on Iris Dataset (Binary)
- Linear kernel SVM with 2D decision boundary visualization

### Q3 — SVM on Moon Dataset (Multi-kernel)
- Synthetic Moon dataset with multiple kernels: linear, poly, RBF
- Visualizes support vectors and decision boundaries per kernel
- Hyperparameter tuning: C and gamma via `RandomizedSearchCV`
- Effect of gamma values on margin width

## Tech Stack
- Python, NumPy, scikit-learn, OpenCV, Matplotlib

## Key Files
- `B22AI045_all.ipynb` — complete notebook
- `B22AI045_prob1.py` — K-Means image compression
- `B22AI045_prob2.py` — SVM experiments
- `B22AI045_Report.pdf` — assignment report

## Run
```bash
jupyter notebook B22AI045_all.ipynb
```

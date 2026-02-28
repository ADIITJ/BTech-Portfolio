# Machine Learning for Economics — Assignment 4: Treatment Effect Estimation (ATE/CATE)

**Subject:** Machine Learning for Economics (Y4)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Problem
Estimate causal treatment effects using modern econometric and ML methods:
- **ATE (Average Treatment Effect):** Population-level causal impact
- **CATE (Conditional ATE):** Heterogeneous effects across subgroups using meta-learners

## Tech Stack
- Python, causalml, econometrics libraries, LaTeX (report)

## Key Files
- `analysis.ipynb` — ATE/CATE estimation pipeline
- `sim_health.csv` — simulated health intervention dataset
- `report.tex` — LaTeX writeup
- `MLE_4.pdf` — assignment specification

## Run
```bash
pip install causalml pandas numpy jupyter
jupyter notebook analysis.ipynb
```

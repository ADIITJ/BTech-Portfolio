# Machine Learning for Economics — Assignment 3: Causal Inference (DiD & RDD)

**Subject:** Machine Learning for Economics (Y4)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Problem
Causal inference using quasi-experimental methods:
- **Difference-in-Differences (DiD):** Estimates treatment effect by comparing pre/post changes across treatment and control groups
- **Regression Discontinuity Design (RDD):** Exploits sharp cutoffs in assignment rules to identify local average treatment effects

## Tech Stack
- Python, pandas, NumPy, statsmodels

## Key Files
- `AIL7310_Assignment3_Analysis.ipynb` — full analysis
- `did_data.csv` — DiD dataset
- `rdd_data.csv` — RDD dataset
- `results/` — output figures
- `Assignment 3.pdf` — problem specification

## Run
```bash
pip install pandas numpy statsmodels jupyter
jupyter notebook AIL7310_Assignment3_Analysis.ipynb
```

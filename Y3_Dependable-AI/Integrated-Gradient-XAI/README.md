# Dependable AI — Integrated Gradients (Explainable AI)

**Subject:** Dependable AI (Y3)
**Institution:** IIT Jodhpur, Dept. of CSE & AI

## Problem
Explainability for multi-label classification using the Integrated Gradients attribution method. Extends AnnexML with post-hoc explanations: for any prediction, identifies which input features contributed most, and visualizes attribution scores.

## Tech Stack
- C++, Python
- NLTK (text preprocessing)
- Matplotlib (attribution visualization)

## Pipeline
```
Train (AnnexML) → Predict → Evaluate → Explain (Integrated Gradients) → Visualize
```

## Key Files
- `scripts/` — attribution computation
- `Integrated-Grad/` — visualization outputs
- `Integrated-Grad.pdf` — methodology reference

## Reference
Axiomatic Attribution for Deep Networks (Sundararajan et al., 2017).

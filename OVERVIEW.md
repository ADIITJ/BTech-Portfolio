# BTech Portfolio — Operational Overview for Claude

**Owner:** Aashish Date (B22AI045) · IIT Jodhpur, CSE-AI
**Portfolio root:** `BTech-Portfolio/` inside `/Users/ashishdate/Documents/IITJ/`
**GitHub profile:** https://github.com/ADIITJ

This document tells Claude exactly how to add, update, or manage projects in this portfolio. Read this before making any changes.

---

## 1. Folder Naming Convention

```
Y{N}_{Subject-Name}/{SubjectPrefix-Type-Short-Description}/
```

| Token | Meaning | Examples |
|---|---|---|
| `Y{N}` | Academic year (1–4) | `Y2`, `Y3`, `Y4` |
| `{Subject-Name}` | Full subject name, hyphen-separated | `Deep-Learning`, `Computer-Vision`, `PRML`, `PCS-2` |
| `{SubjectPrefix-Type-Description}` | Subject abbreviation + type + topic | `PRML-A4-LDA-Naive-Bayes`, `CV-A1-Camera-Calibration`, `DL-Final-Fashion-Style-Transfer` |

### Type prefixes (in the project folder name)
| Prefix | Meaning |
|---|---|
| `A{N}-` | Numbered assignment (e.g., `A1-`, `A3-`) |
| `Final-` | Course final/capstone project |
| `{SubjectAbbr}-Final-` | Subject-scoped final project (e.g., `DL-Final-`) |
| `BTP_` | B.Tech final year project |
| *(none)* | Standalone/independent project |

### Examples of correct folder names
```
Y2_PCS-2/SHAREKARO-P2P-File-Sharing/
Y2_PRML/PRML-A1-Decision-Trees-Linear-Regression/
Y2_PRML/PRML-A5-KMeans-Clustering-SVM/
Y3_Deep-Learning/DL-Final-Fashion-Style-Transfer/
Y3_Deep-Learning/MNIST-ONNX-CPP-Inference/
Y4_Computer-Vision/CV-A1-Camera-Calibration/
Y4_Computer-Vision/CV-Final-Image-Colorization/
Y4_Machine-Learning-Economics/MLE-A3-DiD-RDD-Causal-Inference/
BTP_TableQA-KnowledgeGraph-Augmentation/BTP_Final/
Independent_NLP-RAG-Implementation/
```

---

## 2. Current Portfolio Structure

```
BTech-Portfolio/
├── README.md                           ← root portfolio (update for every new project)
├── OVERVIEW.md                         ← this file
├── .gitignore                          ← excludes venvs, pycache, checkpoints, DS_Store
│
├── Y2_PCS-2/
│   ├── README.md                       ← subject-level overview
│   └── SHAREKARO-P2P-File-Sharing/     ← GitHub: ADIITJ/SHARE-KARO
│
├── Y2_PRML/
│   ├── README.md                       ← subject-level overview
│   ├── B22AI045.zip                    ← archive
│   ├── PRML-A1-Decision-Trees-Linear-Regression/
│   ├── PRML-A3-Perceptron-PCA-Face-Recognition/
│   ├── PRML-A4-LDA-Naive-Bayes/
│   ├── PRML-A5-KMeans-Clustering-SVM/
│   └── PRML-A6-MLP-Neural-Networks/
│       └── alt-submission/             ← duplicate A6 version, kept for reference
│
├── Y3_Deep-Learning/
│   ├── README.md
│   ├── DL-NLP-Assignments/
│   ├── Detectron2-Object-Detection/
│   ├── MNIST-ONNX-CPP-Inference/       ← has nested .git
│   └── DL-Final-Fashion-Style-Transfer/ ← GitHub: ADIITJ/Fashion-Transfer-main
│
├── Y3_Dependable-AI/
│   ├── README.md
│   ├── AnnexML-Extreme-MultiLabel-Classification/
│   └── Integrated-Gradient-XAI/
│
├── Y3_Data-Engineering/
│   ├── README.md
│   ├── Book-Recommendation-System/     ← has nested .git
│   └── Artwork-Marketplace-Platform/
│
├── Y4_Computer-Vision/
│   ├── README.md
│   ├── CV-A1-Camera-Calibration/
│   ├── CV-A2-Perspective-Warping/
│   ├── CV-A3-Edge-Detection-Feature-Extraction/
│   └── CV-Final-Image-Colorization/    ← has nested .git
│
├── Y4_Computer-Graphics/
│   ├── README.md
│   ├── CG-A1-Geometric-Transformations/
│   ├── CG-A2-2D-Game-Engine-OpenGL/
│   └── CG-A3-GLSL-Shader-Programming/
│
├── Y4_Machine-Learning-Economics/
│   ├── README.md
│   ├── MLE-A1-Housing-Regression/
│   ├── MLE-A2-Credit-Risk-Analysis/
│   ├── MLE-A3-DiD-RDD-Causal-Inference/
│   ├── MLE-A4-Treatment-Effect-ATE-CATE/
│   └── Research-Papers/
│
├── Y4_Autonomous-Systems/
│   ├── README.md
│   ├── Autonomous-A1-Path-Planning/    ← has nested .git inside Assignment_1/
│   └── Autonomous-A3-Particle-Filter-SLAM/
│
├── Y4_Advanced-Machine-Learning/
│   ├── README.md
│   └── Advanced-ML-Coursework/         ← has nested .git inside advanced-ML/
│
├── BTP_TableQA-KnowledgeGraph-Augmentation/
│   ├── BTP_Final/                      ← has nested .git · GitHub: ADIITJ/kg-table-qa
│   └── BTP_v1-TableQA-Intermediate/
│
├── Independent_NLP-RAG-Implementation/ ← has nested .git
│
└── Unsorted/
    ├── ML-Platform-Fullstack/          ← subject unclear, needs review
    ├── HB-Notes/                       ← contents unreviewed
    └── RAG-Techniques.odt              ← misplaced document, review needed
```

---

## 3. Git / .git Rules

**Policy: leave all nested `.git` folders in place.**

The root `BTech-Portfolio/` is its own git repo. Nested `.git` folders are NOT submodules — they are historical repos that were moved in. This is intentional.

> ⚠️ **GitHub display note:** Folders with nested `.git` appear as greyed-out broken submodule links on GitHub. This is cosmetic only — the files are still there and accessible. To fix the appearance, the nested `.git` folders would need to be removed or converted to proper submodules. The current policy keeps them as-is.

### When adding a GitHub-hosted project
**Option A — Reference folder (preferred for already-public repos):**
1. Create the folder at the correct path
2. Write a `README.md` with portfolio context block + GitHub link
3. Do NOT clone source files

**Option B — Clone into portfolio (when user wants code locally):**
```bash
git clone https://github.com/ADIITJ/<repo-name> \
  "BTech-Portfolio/Y{N}_{Subject}/{ProjectType-Description}"
# .git folder stays — do NOT remove it
```

**Option C — Local folder already exists:**
- Move it with `shutil.move()` (Python) — do NOT use `mv` in bash (space issues)
- Leave any nested `.git` as-is

---

## 4. README Rules

### 4a. Every project folder must have a README.md

Template for new projects:
```markdown
# {Subject} — {Project Description}

**Subject:** {Full Course Name} ({Year})
**Institution:** IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/{repo}   ← only if GitHub-hosted

## Problem
[1–3 sentences]

## Tech Stack
- Language, Framework
- Key libraries

## Key Files
- `file.py` — purpose
- `folder/` — contents

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
```

### 4b. If an existing README is already present
Do NOT overwrite. Prepend a portfolio context block:
```markdown
<!-- PORTFOLIO CONTEXT -->
**Subject:** {Course} ({Year}) · IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/{repo}
**Problem:** [one line]
**Tech Stack:** [comma-separated]
<!-- END PORTFOLIO CONTEXT -->

---

{original content unchanged}
```

### 4c. Subject-level README (Y{N}_{Subject}/README.md)
Every subject folder must also have a `README.md` — brief overview + project table:
```markdown
# Y{N} — {Full Subject Name}

**Course:** {Subject} · {Year} Year, IIT Jodhpur
**Focus:** [one sentence on the course theme]

| Folder | Type | Description |
|---|---|---|
| [ProjectFolder](./ProjectFolder/) | Assignment/Final/Lab | Brief description |
```

---

## 5. How to Add a New Project — Step-by-Step

### Step 1 — Identify placement
Determine:
- Year (`Y2`, `Y3`, `Y4`)
- Subject → look up in Section 6 table
- Type: assignment number, final project, or standalone
- Source: local disk or GitHub URL

If year/subject unknown → place in `Unsorted/`, flag to user.

### Step 2 — Create the folder
```python
import os
os.makedirs("BTech-Portfolio/Y{N}_{Subject}/{Prefix-Description}", exist_ok=True)
```
Always use Python for file ops. Never use bash `mv` on paths with spaces.

### Step 3 — Handle source files
- **Local folder:** `shutil.move(src, dst)`, leave .git in place
- **GitHub reference:** skip — README only
- **GitHub clone:** `git clone <url> <dst>`, leave .git in place

### Step 4 — Write READMEs
- **Project README:** use template from 4a/4b
- **Subject README:** update the project table in `Y{N}_{Subject}/README.md`

### Step 5 — Update root README.md (two places)

**Table of Contents row** (add to existing subject row, or new row for new subject):
```markdown
| {N} | [{Subject}](#{anchor}) | Y{N} | Project1, Project2, NewProject |
```

**Subject section table** (add new row):
```markdown
| [Folder-Name](./Y{N}_{Subject}/Folder-Name/) | 1-sentence description | Tech1, Tech2 |
```
Add ⭐ for final projects and production-level work.

### Step 6 — Update Technology Summary
At the bottom of `README.md`, add new languages/frameworks to the relevant row.

---

## 6. Known Subjects and Folder Names

| Course Name | Folder Prefix | Year |
|---|---|---|
| Programming of Computer Systems 2 (PCS-2) | `Y2_PCS-2` | Y2 |
| Pattern Recognition & Machine Learning (PRML) | `Y2_PRML` | Y2 |
| Deep Learning | `Y3_Deep-Learning` | Y3 |
| Dependable AI | `Y3_Dependable-AI` | Y3 |
| Data Engineering | `Y3_Data-Engineering` | Y3 |
| Computer Vision | `Y4_Computer-Vision` | Y4 |
| Computer Graphics | `Y4_Computer-Graphics` | Y4 |
| Machine Learning for Economics (MLE) | `Y4_Machine-Learning-Economics` | Y4 |
| Autonomous Systems | `Y4_Autonomous-Systems` | Y4 |
| Advanced Machine Learning | `Y4_Advanced-Machine-Learning` | Y4 |
| B.Tech Project (BTP) | `BTP_TableQA-KnowledgeGraph-Augmentation` | Y4 |
| Independent / Personal | `Independent_{Description}` | — |
| Unknown / Unsorted | `Unsorted/` | — |

For a new course not in this table: ask the user for the year, then add it.

---

## 7. Owner Information

```
Name:     Aashish Date
Roll No:  B22AI045
GitHub:   https://github.com/ADIITJ
LinkedIn: https://www.linkedin.com/in/atharva-date-a956b6256/
Email:    aashishdate@proliantdatallc.com
```

---

## 8. Root README Section Anchors

| Section | Anchor |
|---|---|
| 1. PCS-2 · Y2 | `#1-programming-of-computer-systems-pcs-2--y2` |
| 2. PRML · Y2 | `#2-pattern-recognition--machine-learning-prml--y2` |
| 3. Deep Learning · Y3 | `#3-deep-learning--y3` |
| 4. Dependable AI · Y3 | `#4-dependable-ai--y3` |
| 5. Data Engineering · Y3 | `#5-data-engineering--y3` |
| 6. Computer Vision · Y4 | `#6-computer-vision--y4` |
| 7. Computer Graphics · Y4 | `#7-computer-graphics--y4` |
| 8. ML for Economics · Y4 | `#8-machine-learning-for-economics--y4` |
| 9. Autonomous Systems · Y4 | `#9-autonomous-systems--y4` |
| 10. Advanced ML · Y4 | `#10-advanced-machine-learning--y4` |
| 11. BTP | `#11-btech-final-project-btp` |
| 12. Independent | `#12-independent-projects` |

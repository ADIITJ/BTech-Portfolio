# BTech Portfolio — Operational Overview for Claude

**Owner:** Aashish Date (B22AI045) · IIT Jodhpur, CSE-AI
**Portfolio root:** `BTech-Portfolio/` inside `/Users/ashishdate/Documents/IITJ/`
**GitHub profile:** https://github.com/ADIITJ

This document tells Claude exactly how to add, update, or manage projects in this portfolio. Read this before making any changes.

---

## 1. Folder Naming Convention

```
Y{N}_{Subject-Name}/{ProjectType-Short-Description}/
```

| Token | Meaning | Examples |
|---|---|---|
| `Y{N}` | Academic year (1–4) | `Y2`, `Y3`, `Y4` |
| `{Subject-Name}` | Full subject name, hyphen-separated | `Deep-Learning`, `Computer-Vision`, `PCS-2` |
| `{ProjectType-Short-Description}` | Type prefix + short description | `DL-Final-Fashion-Style-Transfer`, `CV-A1-Camera-Calibration`, `SHAREKARO-P2P-File-Sharing` |

### ProjectType prefixes
| Prefix | Meaning |
|---|---|
| `A{N}-` | Numbered assignment (e.g., `A1-`, `A3-`) |
| `Final-` or `DL-Final-` | Course final/capstone project |
| `BTP_` | B.Tech final year project |
| *(none)* | Standalone/independent project |

### Examples of correct folder names
```
Y2_PCS-2/SHAREKARO-P2P-File-Sharing/
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
│
├── Y2_PCS-2/
│   └── SHAREKARO-P2P-File-Sharing/     ← GitHub: ADIITJ/SHARE-KARO
│
├── Y3_Deep-Learning/
│   ├── DL-NLP-Assignments/
│   ├── Detectron2-Object-Detection/
│   ├── MNIST-ONNX-CPP-Inference/       ← has nested .git
│   └── DL-Final-Fashion-Style-Transfer/ ← GitHub: ADIITJ/Fashion-Transfer-main
│
├── Y3_Dependable-AI/
│   ├── AnnexML-Extreme-MultiLabel-Classification/
│   └── Integrated-Gradient-XAI/
│
├── Y3_Data-Engineering/
│   ├── Book-Recommendation-System/     ← has nested .git
│   └── Artwork-Marketplace-Platform/
│
├── Y4_Computer-Vision/
│   ├── CV-A1-Camera-Calibration/
│   ├── CV-A2-Perspective-Warping/
│   ├── CV-A3-Edge-Detection-Feature-Extraction/
│   └── CV-Final-Image-Colorization/    ← has nested .git
│
├── Y4_Computer-Graphics/
│   ├── CG-A1-Geometric-Transformations/
│   ├── CG-A2-2D-Game-Engine-OpenGL/
│   └── CG-A3-GLSL-Shader-Programming/
│
├── Y4_Machine-Learning-Economics/
│   ├── MLE-A1-Housing-Regression/
│   ├── MLE-A2-Credit-Risk-Analysis/
│   ├── MLE-A3-DiD-RDD-Causal-Inference/
│   ├── MLE-A4-Treatment-Effect-ATE-CATE/
│   └── Research-Papers/
│
├── Y4_Autonomous-Systems/
│   ├── Autonomous-A1-Path-Planning/    ← has nested .git inside Assignment_1/
│   └── Autonomous-A3-Particle-Filter-SLAM/
│
├── Y4_Advanced-Machine-Learning/
│   └── Advanced-ML-Coursework/         ← has nested .git inside advanced-ML/
│
├── BTP_TableQA-KnowledgeGraph-Augmentation/
│   ├── BTP_Final/                      ← has nested .git · GitHub: ADIITJ/kg-table-qa
│   └── BTP_v1-TableQA-Intermediate/
│
├── Independent_NLP-RAG-Implementation/ ← has nested .git · GitHub: ADIITJ/RAG-implementation
│
└── Unsorted/
    ├── ML-Platform-Fullstack/          ← subject unclear, needs review
    └── HB-Notes/                       ← contents unreviewed
```

---

## 3. Git / .git Rules

**Policy: leave all nested `.git` folders in place.**

The root `BTech-Portfolio/` is its own git repo. Nested `.git` folders are NOT submodules — they are just historical repos that were moved in. This is intentional.

### When adding a GitHub-hosted project
There are two valid approaches:

**Option A — Reference folder (no clone, preferred for already-deployed repos):**
1. Create the folder at the correct path
2. Write a `README.md` with a portfolio context block + link to the GitHub repo
3. Do NOT clone or copy source files — the GitHub link is the canonical source

**Option B — Clone into portfolio (when user wants code locally):**
```bash
git clone https://github.com/ADIITJ/<repo-name> \
  "BTech-Portfolio/Y{N}_{Subject}/{ProjectType-Description}"
# .git folder stays — do NOT remove it
```

**Option C — Local folder already exists:**
- Move it with `shutil.move()` (Python) to avoid shell quoting issues
- Leave any nested `.git` as-is

---

## 4. README Rules for Each Project Folder

Every project folder **must** have a `README.md`. Use this template:

```markdown
# {Subject} — {Project Description}

**Subject:** {Full Course Name} ({Year, e.g. Y3})
**Institution:** IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/{repo-name}   ← only if GitHub-hosted

## Problem
[1–3 sentences: what problem does this solve?]

## Tech Stack
- [Framework / language]
- [Key library]

## Key Files / Structure
- `file.py` — what it does
- `folder/` — what it contains

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
```

### Special cases

**If the project already has a detailed README** (e.g., was cloned from GitHub):
- Do NOT overwrite it
- Prepend a portfolio context block:
```markdown
<!-- PORTFOLIO CONTEXT -->
**Subject:** {Course} ({Year}) · IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/{repo}
**Problem:** [one line summary]
**Tech Stack:** [comma-separated]
<!-- END PORTFOLIO CONTEXT -->

---

{original README content}
```

**If it's a reference-only folder** (code lives on GitHub):
- Write a full standalone README with the GitHub link prominently shown
- Include the pipeline/architecture, tech stack, and how-to-run pointing to the GitHub repo

---

## 5. How to Add a New Project — Step-by-Step

When the user says "add this project to my portfolio", follow these steps exactly:

### Step 1 — Identify the target folder path
Ask (or infer from context):
- What year? (`Y2`, `Y3`, `Y4`)
- What subject/course? → maps to `{Subject-Name}` in the convention
- Is it an assignment (A1, A2...) or a final project or standalone?
- Is the code local (on disk) or on GitHub?

If you cannot determine the year or subject, place it in `Unsorted/` and note it.

### Step 2 — Create the folder
```python
import os
os.makedirs("BTech-Portfolio/Y{N}_{Subject}/{ProjectType-Description}", exist_ok=True)
```
Use Python for all file operations to avoid shell quoting issues with spaces.

### Step 3 — Handle the source
- **Local folder:** Use `shutil.move(src, dst)` — do NOT use `mv` in bash for paths with spaces
- **GitHub repo (reference):** Skip — just write the README with the link
- **GitHub repo (clone):** `git clone <url> <destination>` — leave `.git` in place

### Step 4 — Write README.md
- If existing README exists: read it first, then prepend the portfolio context block
- If no README: write one from scratch using the template in Section 4
- Extract tech stack and problem from: code files, existing README, or GitHub description

### Step 5 — Update root README.md
The root `README.md` has two places to update:

**5a. Table of Contents** (top of file):
```markdown
| {N} | [{Subject}](#{anchor}) | Y{year} | {comma-separated project names} |
```
- Anchor format: `#N-subject-name--yN` (lowercase, spaces→hyphens, special chars removed)
- If the subject section already exists: just add the project name to the Projects column
- If new subject: add a new row AND a new section below

**5b. Subject section** (body of file):
- If the subject section already exists: add a new row to its project table
- If new subject: add a new `## N. Subject · YN` section with the table

**Table row format:**
```markdown
| [Folder-Name](./Y{N}_{Subject}/Folder-Name/) | Description (1 sentence) | Tech1, Tech2 |
```
Add ⭐ after the folder link for final projects or production-level work.

### Step 6 — Update Technology Summary table
At the bottom of `README.md`, add any new languages/frameworks to the relevant row.

---

## 6. Known Subjects and Their Folder Names

| Course Name | Folder Prefix | Year |
|---|---|---|
| Programming of Computer Systems 2 (PCS-2) | `Y2_PCS-2` | Y2 |
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

If the user mentions a course not in this table, ask for the year and add it as a new folder.

---

## 7. Owner Information (for READMEs and commits)

```
Name:     Aashish Date
Roll No:  B22AI045
GitHub:   https://github.com/ADIITJ
LinkedIn: https://www.linkedin.com/in/atharva-date-a956b6256/
Email:    aashishdate@proliantdatallc.com
```

---

## 8. Root README Section Anchor Reference

Use these exact anchors when updating the ToC:

| Section | Anchor |
|---|---|
| 1. PCS-2 · Y2 | `#1-programming-of-computer-systems-pcs-2--y2` |
| 2. Deep Learning · Y3 | `#2-deep-learning--y3` |
| 3. Dependable AI · Y3 | `#3-dependable-ai--y3` |
| 4. Data Engineering · Y3 | `#4-data-engineering--y3` |
| 5. Computer Vision · Y4 | `#5-computer-vision--y4` |
| 6. Computer Graphics · Y4 | `#6-computer-graphics--y4` |
| 7. ML for Economics · Y4 | `#7-machine-learning-for-economics--y4` |
| 8. Autonomous Systems · Y4 | `#8-autonomous-systems--y4` |
| 9. Advanced ML · Y4 | `#9-advanced-machine-learning--y4` |
| 10. BTP | `#10-btech-final-project-btp` |
| 11. Independent | `#11-independent-projects` |

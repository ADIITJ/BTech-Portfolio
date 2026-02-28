# BTech Portfolio — Aashish Date (B22AI045)

**Institution:** Indian Institute of Technology Jodhpur
**Department:** Computer Science & Engineering — Artificial Intelligence
**GitHub:** [ADIITJ](https://github.com/ADIITJ) · **LinkedIn:** [Atharva Date](https://www.linkedin.com/in/atharva-date-a956b6256/)

---

## Table of Contents

| # | Subject | Year | Projects |
|---|---|---|---|
| 1 | [Programming of Computer Systems (PCS-2)](#1-programming-of-computer-systems-pcs-2--y2) | Y2 | SHAREKARO P2P Platform |
| 2 | [Deep Learning](#2-deep-learning--y3) | Y3 | NLP Assignments, Detectron2, MNIST ONNX, Fashion Style Transfer |
| 3 | [Dependable AI](#3-dependable-ai--y3) | Y3 | AnnexML, Integrated Gradients XAI |
| 4 | [Data Engineering](#4-data-engineering--y3) | Y3 | Book Recommender, Artwork Marketplace |
| 5 | [Computer Vision](#5-computer-vision--y4) | Y4 | Calibration, Warping, Edge Detection, Colorization |
| 6 | [Computer Graphics](#6-computer-graphics--y4) | Y4 | Transformations, 2D Game Engine, GLSL Shaders |
| 7 | [Machine Learning for Economics](#7-machine-learning-for-economics--y4) | Y4 | Regression, Credit Risk, Causal Inference, ATE/CATE |
| 8 | [Autonomous Systems](#8-autonomous-systems--y4) | Y4 | Path Planning, Particle Filter SLAM |
| 9 | [Advanced Machine Learning](#9-advanced-machine-learning--y4) | Y4 | Coursework |
| 10 | [B.Tech Final Project (BTP)](#10-btech-final-project-btp) | Y4 | Table QA + Knowledge Graph Augmentation |
| 11 | [Independent Projects](#11-independent-projects) | — | RAG Implementation |

---

## 1. Programming of Computer Systems (PCS-2) · Y2

**Course focus:** Systems programming, networking, low-level communication protocols.

| Project | Description | Tech |
|---|---|---|
| [SHAREKARO-P2P-File-Sharing](./Y2_PCS-2/SHAREKARO-P2P-File-Sharing/) ⭐ | P2P messaging and file-transfer platform over Wi-Fi with TCP integrity checks and real-time network monitoring. No central server. | Python, TCP sockets, psutil |

> Source: [github.com/ADIITJ/SHARE-KARO](https://github.com/ADIITJ/SHARE-KARO)

---

## 2. Deep Learning · Y3

**Course focus:** Deep learning architectures, training pipelines, NLP and computer vision applications.

| Project | Description | Tech |
|---|---|---|
| [DL-NLP-Assignments](./Y3_Deep-Learning/DL-NLP-Assignments/) | Coursework covering NLU, NLP task training pipelines | Python, PyTorch, NLTK |
| [Detectron2-Object-Detection](./Y3_Deep-Learning/Detectron2-Object-Detection/) | Object detection using Facebook Research's Detectron2 | Python, Detectron2, PyTorch |
| [MNIST-ONNX-CPP-Inference](./Y3_Deep-Learning/MNIST-ONNX-CPP-Inference/) | Cross-platform digit classifier: trained in Python, inferred in C++ via ONNX Runtime | C++11, ONNX Runtime, OpenCV, CMake |
| [DL-Final-Fashion-Style-Transfer](./Y3_Deep-Learning/DL-Final-Fashion-Style-Transfer/) ⭐ | **Course final project.** Cloth segmentation + saliency-weighted neural style transfer for virtual fashion try-on | Python, PyTorch, U-2-Net, VGG-19 |

> Fashion Transfer source: [github.com/ADIITJ/Fashion-Transfer-main](https://github.com/ADIITJ/Fashion-Transfer-main)

---

## 3. Dependable AI · Y3

**Course focus:** Robustness, scalability, and explainability in ML systems.

| Project | Description | Tech |
|---|---|---|
| [AnnexML-Extreme-MultiLabel-Classification](./Y3_Dependable-AI/AnnexML-Extreme-MultiLabel-Classification/) | KDD 2017 paper implementation: Approximate NN search for 10⁴–10⁶ label spaces | C++11, OpenMP, Python |
| [Integrated-Gradient-XAI](./Y3_Dependable-AI/Integrated-Gradient-XAI/) | Post-hoc attribution explanations using Integrated Gradients on AnnexML predictions | C++, Python, NLTK |

---

## 4. Data Engineering · Y3

**Course focus:** Data pipelines, web applications, containerized deployment.

| Project | Description | Tech |
|---|---|---|
| [Book-Recommendation-System](./Y3_Data-Engineering/Book-Recommendation-System/) | ML-driven book recommendation web app with MongoDB backend and Docker deployment | Python, Flask, MongoDB, Docker |
| [Artwork-Marketplace-Platform](./Y3_Data-Engineering/Artwork-Marketplace-Platform/) | Full-stack art marketplace with user auth, artwork management, and synthetic data generation | Python, Streamlit/Flask, SQLite |

---

## 5. Computer Vision · Y4

**Course focus:** Image formation, geometric transformations, feature extraction, deep learning for vision.

| Project | Description | Tech |
|---|---|---|
| [CV-A1-Camera-Calibration](./Y4_Computer-Vision/CV-A1-Camera-Calibration/) | Intrinsic/extrinsic parameter estimation from checkerboard patterns (Zhang 2000) | Python, OpenCV |
| [CV-A2-Perspective-Warping](./Y4_Computer-Vision/CV-A2-Perspective-Warping/) | Homographic projection and perspective warp transforms | Python, OpenCV |
| [CV-A3-Edge-Detection-Feature-Extraction](./Y4_Computer-Vision/CV-A3-Edge-Detection-Feature-Extraction/) | Spatial filtering, Sobel/Canny edge detection, local feature extraction | Python, NumPy, OpenCV |
| [CV-Final-Image-Colorization](./Y4_Computer-Vision/CV-Final-Image-Colorization/) ⭐ | Automatic image colorization with **novel SPCR evaluation metric**. ResNet-18 encoder, 313-bin Lab classification. 92%+ quality scores | Python, PyTorch, ResNet-18, DeepLabV3 |

---

## 6. Computer Graphics · Y4

**Course focus:** OpenGL rendering pipeline, transformations, shader programming, real-time graphics.

| Project | Description | Tech |
|---|---|---|
| [CG-A1-Geometric-Transformations](./Y4_Computer-Graphics/CG-A1-Geometric-Transformations/) | 2D/3D transformation matrices and camera model fundamentals | Python, OpenCV |
| [CG-A2-2D-Game-Engine-OpenGL](./Y4_Computer-Graphics/CG-A2-2D-Game-Engine-OpenGL/) | Interactive 2D game engine with real-time ImGUI parameter panels | Python, PyOpenGL, ImGUI, GLFW |
| [CG-A3-GLSL-Shader-Programming](./Y4_Computer-Graphics/CG-A3-GLSL-Shader-Programming/) | 3D stacker game implementing programmable vertex/fragment shaders from scratch | Python, PyOpenGL, GLSL |

---

## 7. Machine Learning for Economics · Y4

**Course focus:** Econometrics, causal inference, treatment effect estimation, predictive modeling.

| Project | Description | Tech |
|---|---|---|
| [MLE-A1-Housing-Regression](./Y4_Machine-Learning-Economics/MLE-A1-Housing-Regression/) | EDA and regression modeling on housing price data | Python, pandas, scikit-learn |
| [MLE-A2-Credit-Risk-Analysis](./Y4_Machine-Learning-Economics/MLE-A2-Credit-Risk-Analysis/) | Binary classification for loan default prediction | Python, scikit-learn |
| [MLE-A3-DiD-RDD-Causal-Inference](./Y4_Machine-Learning-Economics/MLE-A3-DiD-RDD-Causal-Inference/) | Difference-in-Differences and Regression Discontinuity Design analysis | Python, statsmodels |
| [MLE-A4-Treatment-Effect-ATE-CATE](./Y4_Machine-Learning-Economics/MLE-A4-Treatment-Effect-ATE-CATE/) | Average and Conditional Average Treatment Effect estimation via meta-learners | Python, causalml |
| [Research-Papers](./Y4_Machine-Learning-Economics/Research-Papers/) | 13 papers on Google Trends, sentiment analysis, and macroeconomic ML | — |

---

## 8. Autonomous Systems · Y4

**Course focus:** Probabilistic robotics, path planning, localization, SLAM.

| Project | Description | Tech |
|---|---|---|
| [Autonomous-A1-Path-Planning](./Y4_Autonomous-Systems/Autonomous-A1-Path-Planning/) | Search/sampling-based robot path planning algorithms | Python |
| [Autonomous-A3-Particle-Filter-SLAM](./Y4_Autonomous-Systems/Autonomous-A3-Particle-Filter-SLAM/) | Particle filter for probabilistic robot localization (Monte Carlo approach) | Python, NumPy |

---

## 9. Advanced Machine Learning · Y4

**Course focus:** Advanced ML theory and methods beyond standard coursework.

| Project | Description | Tech |
|---|---|---|
| [Advanced-ML-Coursework](./Y4_Advanced-Machine-Learning/Advanced-ML-Coursework/) | Course repository with assignments covering advanced ML topics | Python |

---

## 10. B.Tech Final Project (BTP)

**Domain:** Natural Language Processing / Knowledge Graphs / Multi-Agent Systems

| Project | Description | Tech |
|---|---|---|
| [BTP_Final](./BTP_TableQA-KnowledgeGraph-Augmentation/BTP_Final/) ⭐ | **Production system.** 6-agent hierarchical architecture answers questions over incomplete tables by augmenting with Wikidata knowledge. **92.9% success rate vs 28.6% baseline (225% improvement)** | Python, Pydantic, asyncio, NetworkX, Gemini API, Wikidata SPARQL, Redis |
| [BTP_v1-TableQA-Intermediate](./BTP_TableQA-KnowledgeGraph-Augmentation/BTP_v1-TableQA-Intermediate/) | Earlier iteration of the same system, retained for development history | Python |

> Source: [github.com/ADIITJ/kg-table-qa](https://github.com/ADIITJ/kg-table-qa)

### BTP System Architecture
```
Question + Incomplete Table
        │
        ▼
┌─────────────────────────┐
│   Orchestrator Agent     │  ← intelligent routing
└────────────┬────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
Gap Detection    Entity Linking
Agent            Agent (Wikidata QIDs)
    │                 │
    └────────┬────────┘
             ▼
      KG Retrieval Agent
      (ensemble voting)
             │
             ▼
      Graph Builder Agent
      (hybrid NetworkX)
             │
             ▼
      Reasoning Agent
      (multi-path BFS)
             │
             ▼
         Answer
```

---

## 11. Independent Projects

| Project | Description | Tech |
|---|---|---|
| [Independent_NLP-RAG-Implementation](./Independent_NLP-RAG-Implementation/) | RAG prompt generation system with TF-IDF + sentence-transformer retrieval, chunking, and sentiment analysis | Python, NLTK, scikit-learn, sentence-transformers |

---

## Technology Summary

| Domain | Technologies |
|---|---|
| ML / DL | PyTorch, TensorFlow, scikit-learn, Detectron2, HuggingFace |
| NLP | NLTK, sentence-transformers, Gemini API, RAG, SPARQL/Wikidata |
| Computer Vision | OpenCV, NumPy, ResNet-18, DeepLabV3, VGG-19, U-2-Net |
| Computer Graphics | PyOpenGL, GLSL, ImGUI, GLFW, pyrr |
| Systems / Networking | C++11, ONNX Runtime, CMake, OpenMP, TCP sockets, psutil |
| Web / Data | Python (Flask, Streamlit), MongoDB, SQLite, Docker |
| Infrastructure | asyncio, Pydantic, NetworkX, Redis, Git |
| Causal Inference | causalml, statsmodels, DiD, RDD, meta-learners |

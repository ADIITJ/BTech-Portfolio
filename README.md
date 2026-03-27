# Atharva Date — B.Tech Portfolio

**B.Tech · Computer Science & Artificial Intelligence · IIT Jodhpur · 2022–2026**

[GitHub @ADIITJ](https://github.com/ADIITJ) · [LinkedIn](https://www.linkedin.com/in/atharva-date-a956b6256/) · B22AI045

---

4 years · 12 courses · 30+ projects spanning machine learning, computer vision, graphics, NLP, robotics, and systems programming.

---

## Featured Work

The strongest projects from across the degree:

| Project | One Line | Achievement |
|---|---|---|
| [BTP: TableQA + Knowledge Graphs](./BTP_TableQA-KnowledgeGraph-Augmentation/BTP_Final/) | 6-agent system answers questions over incomplete tables using Wikidata | **92.9% success rate vs 28.6% baseline — 225% improvement** |
| [First-Aid AI](./Y4_Smart-Product-Design/First-Aid-AI/) | Real-time sports injury detection and automated emergency response at the edge | YOLOv11n → LLM analysis → MQTT dispatch in 4 stages |
| [Image Colorization](./Y4_Computer-Vision/CV-Final-Image-Colorization/) | Automatic photo colorization with a novel perceptual evaluation metric (SPCR) | 92%+ quality score, ResNet-18 encoder |
| [Fashion Style Transfer](./Y3_Deep-Learning/DL-Final-Fashion-Style-Transfer/) | Cloth segmentation + saliency-weighted neural style transfer for virtual try-on | End-to-end pipeline, U-2-Net + VGG-19 |
| [AnnexML — Extreme Multi-Label](./Y3_Dependable-AI/AnnexML-Extreme-MultiLabel-Classification/) | KDD 2017 paper re-implementation: approx. nearest-neighbour search at 10⁶-label scale | C++11 + OpenMP, full paper replication |
| [SHARE-KARO](./Y2_PCS-2/SHAREKARO-P2P-File-Sharing/) | Serverless P2P file-sharing and messaging over local Wi-Fi | TCP + ACK integrity, no central server |

---

## Year 2

### Programming of Computer Systems (PCS-2)

Systems programming, networking, and low-level communication protocols.

| Project | Description | Tech |
|---|---|---|
| [SHAREKARO-P2P-File-Sharing](./Y2_PCS-2/SHAREKARO-P2P-File-Sharing/) ⭐ | P2P messaging and file transfer over Wi-Fi — real-time network monitoring, no central server | Python, TCP sockets, psutil |

> Source: [github.com/ADIITJ/SHARE-KARO](https://github.com/ADIITJ/SHARE-KARO)

---

### Pattern Recognition & Machine Learning (PRML)

Classical ML algorithms implemented from scratch — decision theory, linear models, kernel methods, neural networks.

> Assignment 2 is not present in this repository.

| Assignment | Topic | Algorithms |
|---|---|---|
| [A1 — Decision Trees & Linear Regression](./Y2_PRML/PRML-A1-Decision-Trees-Linear-Regression/) | Supervised Learning Basics | Decision Tree (scratch), Simple/Multiple LR, Gradient Descent |
| [A3 — Perceptron & Face Recognition](./Y2_PRML/PRML-A3-Perceptron-PCA-Face-Recognition/) | Linear Classifiers & Dim. Reduction | Perceptron, PCA, KNN on LFW dataset |
| [A4 — LDA & Naive Bayes](./Y2_PRML/PRML-A4-LDA-Naive-Bayes/) | Discriminant Analysis | LDA (Sw/Sb/eigenvectors from scratch), Gaussian NB, Multinomial NB |
| [A5 — K-Means & SVM](./Y2_PRML/PRML-A5-KMeans-Clustering-SVM/) | Clustering & Kernels | K-Means image compression (custom), SVM with Linear/Poly/RBF kernels |
| [A6 — MLP & Neural Networks](./Y2_PRML/PRML-A6-MLP-Neural-Networks/) | Deep Foundations | MLP with/without ReLU on MNIST, Backprop, Adam optimiser |

---

## Year 3

### Deep Learning

Deep learning architectures, training pipelines, NLP and computer vision applications.

| Project | Type | Description | Tech |
|---|---|---|---|
| [DL-NLP-Assignments](./Y3_Deep-Learning/DL-NLP-Assignments/) | Coursework | NLU and NLP task training pipelines | PyTorch, NLTK |
| [Detectron2-Object-Detection](./Y3_Deep-Learning/Detectron2-Object-Detection/) | Lab | Object detection using Facebook Research's Detectron2 | Python, Detectron2 |
| [MNIST-ONNX-CPP-Inference](./Y3_Deep-Learning/MNIST-ONNX-CPP-Inference/) | Lab | Train in Python, infer in C++ via ONNX Runtime — cross-platform digit classifier | C++11, ONNX Runtime, CMake |
| [DL-Final — Fashion Style Transfer](./Y3_Deep-Learning/DL-Final-Fashion-Style-Transfer/) ⭐ | **Final Project** | Cloth segmentation + saliency-weighted neural style transfer for virtual fashion try-on | PyTorch, U-2-Net, VGG-19 |

> Source: [github.com/ADIITJ/Fashion-Transfer-main](https://github.com/ADIITJ/Fashion-Transfer-main)

<details>
<summary><strong>Lab Notebooks (9 labs)</strong> — click to expand</summary>

9 progressive hands-on sessions in [DL-Labs](./Y3_Deep-Learning/DL-Labs/):

Labs 1–9 covering: foundations → CNNs → RNNs → attention → advanced training techniques. See the [DL-Labs README](./Y3_Deep-Learning/DL-Labs/README.md) for details.

</details>

<details>
<summary><strong>Lecture Notes</strong> — click to expand</summary>

Full compiled class notes (~31 MB PDF) in [DL-Lectures](./Y3_Deep-Learning/DL-Lectures/).

</details>

---

### Dependable AI

Scalability, robustness, and explainability in ML systems.

| Project | Description | Tech |
|---|---|---|
| [AnnexML — Extreme Multi-Label Classification](./Y3_Dependable-AI/AnnexML-Extreme-MultiLabel-Classification/) ⭐ | Re-implementation of KDD 2017 paper: approx. nearest-neighbour search for label spaces with 10⁴–10⁶ classes | C++11, OpenMP, Python |
| [Integrated Gradient XAI](./Y3_Dependable-AI/Integrated-Gradient-XAI/) | Post-hoc attribution explanations using Integrated Gradients on AnnexML predictions | C++, Python, NLTK |

---

### Data Engineering

Data pipelines, containerised web applications, database integration.

| Project | Description | Tech |
|---|---|---|
| [Book Recommendation System](./Y3_Data-Engineering/Book-Recommendation-System/) | ML-driven book recommendation web app with MongoDB backend and Docker deployment | Flask, MongoDB, Docker |
| [Artwork Marketplace Platform](./Y3_Data-Engineering/Artwork-Marketplace-Platform/) | Full-stack art marketplace with user auth, artwork management, and synthetic data generation | Streamlit/Flask, SQLite |

---

## Year 4

### Computer Vision

Image formation, camera geometry, feature extraction, and deep learning for vision.

| Project | Type | Description | Tech |
|---|---|---|---|
| [CV-A1 — Camera Calibration](./Y4_Computer-Vision/CV-A1-Camera-Calibration/) | Assignment | Intrinsic/extrinsic parameter estimation from checkerboard patterns (Zhang 2000) | Python, OpenCV |
| [CV-A2 — Perspective Warping](./Y4_Computer-Vision/CV-A2-Perspective-Warping/) | Assignment | Homographic projection and perspective warp transforms | Python, OpenCV |
| [CV-A3 — Edge Detection & Feature Extraction](./Y4_Computer-Vision/CV-A3-Edge-Detection-Feature-Extraction/) | Assignment | Spatial filtering, Sobel/Canny edge detection, local feature extraction | Python, NumPy, OpenCV |
| [CV-Final — Image Colorization](./Y4_Computer-Vision/CV-Final-Image-Colorization/) ⭐ | **Final Project** | Automatic image colorization with novel SPCR metric. 92%+ quality score | PyTorch, ResNet-18, DeepLabV3 |

<details>
<summary><strong>Lecture Slides (19 PDFs)</strong> — click to expand</summary>

All slides in [CV Lectures folder](./Y4_Computer-Vision/Lectures/):

| Lecture | Topic |
|---|---|
| 01 | Introduction & Course Overview |
| 03 | Spatial Filtering |
| 04 | Frequency Domain Filtering |
| 05 | Image Transformations |
| 06 | Camera Model |
| 07 | Projective Geometry & Camera Calibration |
| 08 | Two-View Epipolar Geometry |
| 10–12 | Stereo Vision |
| 13–14 | Structure from Motion (SfM) |
| 15–16 | Feature Detection & Description |
| 18 | Optical Flow & Feature Tracking |
| 19–20 | Image Segmentation — Normalized Cut |
| 20–21 | Image Segmentation — Graph Cut |
| 22 | Introduction to Neural Networks (for CV) |
| 26–27 | Deep Detection & Recognition |
| 30 | Deep Segmentation |
| 31 | Deep Video Understanding |
| 32 | GNN for Computer Vision |
| 35 | ICP (Iterative Closest Point) |

</details>

---

### Computer Graphics

OpenGL rendering pipeline, geometric transformations, shader programming, real-time graphics.

| Project | Type | Description | Tech |
|---|---|---|---|
| [CG-A1 — Geometric Transformations](./Y4_Computer-Graphics/CG-A1-Geometric-Transformations/) | Assignment | 2D/3D transformation matrices and camera model fundamentals | Python, OpenCV |
| [CG-A2 — 2D Game Engine (OpenGL)](./Y4_Computer-Graphics/CG-A2-2D-Game-Engine-OpenGL/) | Assignment | Interactive 2D game engine with real-time ImGUI parameter panels | PyOpenGL, ImGUI, GLFW |
| [CG-A3 — GLSL Shader Programming](./Y4_Computer-Graphics/CG-A3-GLSL-Shader-Programming/) | Assignment | 3D stacker game with programmable vertex and fragment shaders | PyOpenGL, GLSL |

<details>
<summary><strong>Lecture Slides (20 PDFs)</strong> — click to expand</summary>

All slides in [CG Lectures folder](./Y4_Computer-Graphics/Lectures/):

| Lecture | Topic |
|---|---|
| Lec 01 | Introduction to the Course |
| Lec 02 | Graphics Pipeline |
| Lec 03–04 | Graphics Primitives (Points, Shapes) |
| Lec 05 | Order of Transformations |
| Lec 06 | Arbitrary Rotations |
| Lec 07 | 3D Graphics Pipeline |
| Lec 08 | Projection Transformations |
| Lec 09 | Viewport Transformations |
| Lec 10 | Primitive Pipeline |
| Lec 11 | Clipping & Culling |
| Lec 12 | Rasterization |
| Lec 14–15 | Visibility & List Priority Algorithms |
| Lec 16 | Colour & Illumination |
| Lec 17 | Lighting & Shading |
| Lec 18 | Shadows & Texture Mapping |
| Lec 20 | Recursive Ray Tracing |
| Lec 21 | Curve & Surface Representation |
| Tut 02 | PyOpenGL 2D World (Tutorial) |

</details>

---

### Machine Learning for Economics

Applied ML for economic analysis — regression, causal inference, and treatment effect estimation.

| Assignment | Topic | Methods |
|---|---|---|
| [A1 — Housing Regression](./Y4_Machine-Learning-Economics/MLE-A1-Housing-Regression/) | Predictive Modelling | EDA, linear/polynomial regression |
| [A2 — Credit Risk Analysis](./Y4_Machine-Learning-Economics/MLE-A2-Credit-Risk-Analysis/) | Classification | Logistic regression, decision trees, model evaluation |
| [A3 — Causal Inference (DiD & RDD)](./Y4_Machine-Learning-Economics/MLE-A3-DiD-RDD-Causal-Inference/) | Causal Inference | Difference-in-Differences, Regression Discontinuity Design |
| [A4 — Treatment Effects (ATE/CATE)](./Y4_Machine-Learning-Economics/MLE-A4-Treatment-Effect-ATE-CATE/) | Treatment Effect Estimation | Meta-learners (S/T/X-learner), ATE & CATE |
| [Research Papers](./Y4_Machine-Learning-Economics/Research-Papers/) | Reference | 13 papers on Google Trends, sentiment analysis, macroeconomic ML |

---

### Autonomous Systems

Probabilistic robotics, motion planning, localization, and SLAM.

| Assignment | Description | Tech |
|---|---|---|
| [A1 — Path Planning](./Y4_Autonomous-Systems/Autonomous-A1-Path-Planning/) | Search and sampling-based robot path planning (A*, RRT, etc.) | Python |
| [A3 — Particle Filter SLAM](./Y4_Autonomous-Systems/Autonomous-A3-Particle-Filter-SLAM/) | Monte Carlo particle filter for probabilistic robot localization | Python, NumPy |

---

### Advanced Machine Learning

Theoretical depth beyond standard coursework — Gaussian Processes, generative models, RL.

| Folder | Description |
|---|---|
| [Advanced-ML-Coursework](./Y4_Advanced-Machine-Learning/Advanced-ML-Coursework/) | Course repository with assignments |

<details>
<summary><strong>Key Papers & Lecture Notes (10 items)</strong> — click to expand</summary>

All readings in [AML Readings folder](./Y4_Advanced-Machine-Learning/Readings/):

| File | Topic | Type |
|---|---|---|
| `additiveModels-part1.pdf` | Additive Models (Part 1) | Lecture Notes |
| `additiveModels-part2.pdf` | Additive Models (Part 2) | Lecture Notes |
| `GP.pdf` | Gaussian Processes — Overview | Lecture Notes |
| `GP-Regression.pdf` | Gaussian Process Regression | Lecture Notes |
| `GP-classification-2.pdf` | GP Classification (Part 2) | Lecture Notes |
| `GPclassification-3.pdf` | GP Classification (Part 3) | Lecture Notes |
| `diffusion.pdf` | Diffusion Models | Research Paper |
| `discrete_vae.pdf` | Discrete Variational Autoencoders | Research Paper |
| `tsne_visualization.pdf` | t-SNE Visualization | Research Paper |
| `Policy-Gradient.pdf` | Policy Gradient Methods (RL) | Research Paper |

</details>

---

### Natural Language Understanding (NLU)

Semantic representations, sentiment analysis, and deep learning for NLP.

<details>
<summary><strong>Lecture Slides (2 PDFs)</strong> — click to expand</summary>

Slides in [NLU Lectures folder](./Y4_NLU/Lectures/):

| File | Topic |
|---|---|
| `_CSL 7040-DeepLearning-Sentiment-March2025.pdf` | Deep Learning for Sentiment Analysis |
| `Lec-Meaning-Vectors-April2025.pdf` | Word Meaning & Vector Representations |

</details>

---

### Smart Product Design

Embedded intelligence, edge AI, real-world system design and deployment.

| Project | Description | Tech |
|---|---|---|
| [First-Aid AI](./Y4_Smart-Product-Design/First-Aid-AI/) ⭐ | Real-time sports injury detection — 4-stage edge pipeline: YOLO detection → H.264 clip capture → multimodal LLM analysis → MQTT dispatch + SMS | YOLOv11n, OpenCV, Qwen3.5-9B, Gemini, MinIO, ChromaDB, MQTT, SQLite |

---

## B.Tech Final Project (BTP)

**Domain:** Natural Language Processing / Knowledge Graphs / Multi-Agent Systems

| Project | Description | Tech |
|---|---|---|
| [BTP_Final](./BTP_TableQA-KnowledgeGraph-Augmentation/BTP_Final/) ⭐ | **Production system.** 6-agent hierarchy answers questions over incomplete tables by augmenting with Wikidata knowledge. **92.9% success rate vs 28.6% baseline — 225% improvement.** | Python, Pydantic, asyncio, NetworkX, Gemini API, Wikidata SPARQL, Redis |
| [BTP_v1 (Intermediate)](./BTP_TableQA-KnowledgeGraph-Augmentation/BTP_v1-TableQA-Intermediate/) | Earlier iteration — kept for development history | Python |

> Source: [github.com/ADIITJ/kg-table-qa](https://github.com/ADIITJ/kg-table-qa)

<details>
<summary><strong>System Architecture</strong> — click to expand</summary>

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

</details>

---

## Independent

| Project | Description | Tech |
|---|---|---|
| [NLP RAG Implementation](./Independent_NLP-RAG-Implementation/) | RAG prompt generation with TF-IDF + sentence-transformer retrieval, chunking, and sentiment analysis | Python, NLTK, scikit-learn, sentence-transformers |

---

## Skills

| Domain | Technologies |
|---|---|
| ML / DL | PyTorch, TensorFlow, scikit-learn, Detectron2, HuggingFace |
| NLP & LLMs | NLTK, sentence-transformers, Gemini API, RAG, SPARQL/Wikidata |
| Computer Vision | OpenCV, NumPy, ResNet-18, DeepLabV3, VGG-19, U-2-Net |
| Computer Graphics | PyOpenGL, GLSL, ImGUI, GLFW, pyrr |
| Systems | C++11, ONNX Runtime, CMake, OpenMP, TCP sockets |
| Web & Data | Flask, Streamlit, MongoDB, SQLite, Docker |
| Infrastructure | asyncio, Pydantic, NetworkX, Redis, MinIO, MQTT, ChromaDB |
| Causal Inference | causalml, statsmodels, DiD, RDD, meta-learners |
| Edge AI | YOLOv11n, ByteTrack, Qwen3.5-9B, ONNX Runtime |

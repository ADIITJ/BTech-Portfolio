# Deep Learning — Final Project: Style Transfer in Fashion Images

**Subject:** Deep Learning — Final Project (Y3)
**Institution:** IIT Jodhpur, Dept. of CSE & AI
**GitHub:** https://github.com/ADIITJ/Fashion-Transfer-main

## Problem
Deep learning-based style transfer system for fashion images. Enables virtual try-on by applying artistic textures and patterns to clothing regions while preserving the background — combining cloth segmentation, saliency-guided attention, and neural style transfer in a unified pipeline.

## Tech Stack
- Python, PyTorch
- U-2-Net (cloth segmentation + saliency map generation)
- VGG-19 (neural style transfer)
- OpenCV, NumPy

## System Pipeline
```
Input Image
    │
    ▼
U-2-Net Cloth Segmentation
(4 classes: background, upper-body, lower-body, full-body)
    │
    ▼
U-2-Net Saliency Map Generation
(highlights prominent clothing regions)
    │
    ▼
VGG-19 Neural Style Transfer
(applies style to segmented clothing region)
    │
    ▼
Saliency-Based Blending
(seamless merge of stylized cloth + original background)
    │
    ▼
Output Image
```

## Key Features
- Precise cloth segmentation (4-class U-2-Net)
- Saliency-weighted style application — avoids over-stylizing edges
- High-quality artistic style transfer via VGG-19 perceptual loss
- Natural background preservation

## How to Run
```bash
git clone https://github.com/ADIITJ/Fashion-Transfer-main
cd Fashion-Transfer-main
pip install -r requirements.txt
# Download U-2-Net model weights (see repo README)
python main.py --content <input_image> --style <style_image>
```

> Full source code, model weights setup, and results: [github.com/ADIITJ/Fashion-Transfer-main](https://github.com/ADIITJ/Fashion-Transfer-main)

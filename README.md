
# Adaptive Hybrid Quantum-Classical CNN for Pneumonia Detection

A full-stack deep learning project that combines classical convolutional neural networks and quantum circuits to detect pneumonia from chest X-ray images.

---

## Overview

This project implements a hybrid architecture composed of:

- MobileNetV3-Small as a classical feature extractor  
- A variational quantum layer using PennyLane  
- An adaptive fusion gate that learns how much to rely on classical vs quantum outputs  

It also includes:

- A complete training pipeline  
- A FastAPI backend for inference  
- A browser-based frontend for image upload and prediction  

---

## Problem Statement

Binary classification of chest X-ray images into:

- NORMAL  
- PNEUMONIA  

Dataset used:  
Chest X-Ray Pneumonia Dataset (Kaggle)

---
## Working Examples

### 1) Prediction with X-ray image of person with pneumonia

<img width="1440" height="861" alt="image" src="https://github.com/user-attachments/assets/df98fdb0-6cca-47f0-811a-d9274bff4a8c" />

### 2) Prediction with X-ray image of a normal healthy person

<img width="1440" height="861" alt="image" src="https://github.com/user-attachments/assets/77e10526-eb6d-4eec-b106-94ca2b6740a6" />

## Model Architecture

### Pipeline

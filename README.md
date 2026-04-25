
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

The dataset used for training the model in this project is taken from : (https://www.kaggle.com/datasets/riyadhhalmosawi/kermanys-cxr-images-datasets)
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

<img width="1440" height="861" alt="image" src="https://github.com/user-attachments/assets/48dea3a7-145d-429d-a7a2-a565368adc8f" />


### 2) Prediction with X-ray image of a normal healthy person

<img width="1440" height="861" alt="image" src="https://github.com/user-attachments/assets/77e10526-eb6d-4eec-b106-94ca2b6740a6" />

## Model Architecture

### Pipeline

```
Input Image
    ↓
MobileNetV3 Feature Extractor
    ↓
Latent Projection
    ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Classical Head   Quantum Layer
 │               │
 ▼               ▼
Logits (c)       Logits (q)
        ↓
Adaptive Fusion Gate
        ↓
Final Prediction
```

---

### Key Components

**1. Classical Backbone**

* MobileNetV3-Small (pretrained on ImageNet)
* Output feature size: 576

**2. Latent Projection**

* Linear(576 → 128)
* ReLU + Dropout
* Linear(128 → n_qubits)

**3. Quantum Layer**

* PennyLane default.qubit simulator
* Angle embedding (RY rotations)
* Trainable variational layers
* Ring entanglement (CNOT)
* Pauli-Z expectation measurements

**4. Output Heads**

* Classical head: Linear(n_qubits → 2)
* Quantum head: Linear(n_qubits → 2)

**5. Adaptive Fusion**

Final logits are computed as:

```
logits = α * q + (1 - α) * c
```

Where:

* c = classical logits
* q = quantum logits
* α = learned fusion weight

---

## Tech Stack

**Core ML**

* torch
* torchvision
* pennylane

**Backend**

* fastapi
* uvicorn
* python-multipart
* pillow

**Utilities**

* numpy
* tqdm

---

## Project Structure

```
.
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
│
├── app/
│   ├── model.py
│   ├── inference.py
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── checkpoints/
│   └── best_model.pt
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 2. Train Model

Full training:

```bash
python train.py --epochs 8 --batch_size 8 --n_qubits 4 --q_layers 2
```

Quick test:

```bash
python train.py --epochs 1 --batch_size 4 --n_qubits 4 --q_layers 2
```

---

### 3. Run Backend

```bash
uvicorn app.main:app --reload
```

Open in browser:

http://127.0.0.1:8000

---

## API

### GET /health

Returns service status and model readiness.

---

### POST /predict

**Input**

* multipart/form-data
* file field: image

**Output**

```json
{
  "prediction": "NORMAL | PNEUMONIA",
  "confidence": 0.0,
  "probabilities": {
    "NORMAL": 0.0,
    "PNEUMONIA": 0.0,
    "adaptive_quantum_weight": 0.0
  }
}
```

---

## Training Details

* Loss: CrossEntropyLoss
* Optimizer: AdamW
* Backbone initially frozen
* Unfreezing after specified epoch
* Metrics tracked:

  * Loss
  * Accuracy
  * Adaptive fusion weight

---

## Notes on Hardware Compatibility

* Classical components run on available accelerators (MPS, CUDA, or CPU)
* Quantum simulation is executed on CPU for stability

---

## Limitations

* No Grad-CAM or interpretability visualizations
* No experiment tracking integration
* Limited evaluation metrics (no ROC-AUC, F1, etc.)
* Uses quantum simulator, not real hardware

---

## Future Improvements

* Add Grad-CAM visualizations
* Add ROC-AUC, precision, recall, F1
* Integrate TensorBoard or Weights & Biases
* Dockerize the application
* Add API test suite

---

## Academic Notes

For evaluation and reporting:

* Compare baseline CNN vs hybrid model
* Perform ablation on:

  * Number of qubits
  * Number of quantum layers
  * Fusion behavior
* Analyze trade-offs of hybrid architectures


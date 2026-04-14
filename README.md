# Adaptive Hybrid Quantum-Classical CNN for Pneumonia Detection

This repository implements a complete end-to-end project for pneumonia detection from chest X-ray images using a hybrid model that combines:

1. A classical CNN feature extractor (MobileNetV3-small)
2. A variational quantum layer (PennyLane simulator)
3. An adaptive fusion gate that learns how much to trust classical vs quantum logits

It also includes:

1. A training pipeline
2. A FastAPI backend for inference
3. A frontend web interface that explains the project and allows image upload/prediction

## 1. What this project solves

Goal: binary classification of chest X-rays into:

1. NORMAL
2. PNEUMONIA

The model is designed to run on macOS Apple Silicon (M1/M2), where:

1. Classical deep learning uses MPS acceleration when available
2. Quantum simulation runs on CPU for compatibility and stability

## 2. Tech stack and why each tool is used

Core ML and quantum:

1. torch: model training, optimization, tensors, autograd
2. torchvision: MobileNetV3 backbone, image transforms, ImageFolder dataset loading
3. pennylane: quantum circuit definition and simulation

Backend and serving:

1. fastapi: HTTP API endpoints
2. uvicorn: ASGI server
3. python-multipart: file upload support in FastAPI
4. pillow: image decoding and preprocessing support

Utility:

1. numpy: numerical support in dependency chain
2. tqdm: progress bars during epoch loops

Dependency file: requirements.txt

## 3. Repository structure

```text
Final YEAR PROJ/
|- chest_xray/                    # dataset root (already present)
|  |- train/
|  |  |- NORMAL/
|  |  |- PNEUMONIA/
|  |- val/
|  |  |- NORMAL/
|  |  |- PNEUMONIA/
|  |- test/
|     |- NORMAL/
|     |- PNEUMONIA/
|
|- app/
|  |- __init__.py
|  |- model.py                    # hybrid model + quantum layer
|  |- inference.py                # checkpoint loading + prediction logic
|  |- main.py                     # FastAPI app + routes
|
|- frontend/
|  |- index.html                  # project page and upload UI
|  |- styles.css                  # styling
|  |- app.js                      # upload + API call logic
|
|- checkpoints/                   # created during training
|  |- best_model.pt
|
|- train.py                       # training and evaluation pipeline
|- requirements.txt
|- .gitignore
|- README.md
```

## 4. Data requirements and format

This project expects Kermany-style folder structure exactly:

```text
chest_xray/
|- train/
|  |- NORMAL/
|  |- PNEUMONIA/
|- val/
|  |- NORMAL/
|  |- PNEUMONIA/
|- test/
   |- NORMAL/
   |- PNEUMONIA/
```

Important:

1. Folder names are class labels used by ImageFolder
2. The default CLI paths in train.py point to this layout
3. If your dataset path differs, use --train_dir, --val_dir, --test_dir

## 5. Model architecture in detail

Defined in app/model.py.

### 5.1 Classical feature extractor

1. Backbone: torchvision.models.mobilenet_v3_small (pretrained ImageNet weights)
2. Uses backbone.features + adaptive avg pool
3. Output feature vector dimension: 576

### 5.2 Latent projection

The 576-D CNN feature vector is reduced to n_qubits dimensional latent space:

1. Linear(576 -> 128)
2. ReLU
3. Dropout(0.2)
4. Linear(128 -> n_qubits)

### 5.3 Quantum branch

Quantum layer configuration:

1. Device: PennyLane default.qubit simulator
2. Embedding: AngleEmbedding(inputs, rotation=Y)
3. Variational block repeated q_layers times
4. Per-qubit trainable Rot(theta, phi, omega)
5. Entanglement ring via CNOT(q -> q+1 mod n_qubits)
6. Measurement: expectation values of PauliZ for each qubit

This returns an n_qubits vector per sample.

### 5.4 Classical and quantum heads

1. classical_head: Linear(n_qubits -> 2)
2. quantum_head: Linear(n_qubits -> 2)

Each head outputs logits for the two classes.

### 5.5 Adaptive fusion gate

Gate network:

1. Linear(n_qubits -> 16)
2. ReLU
3. Linear(16 -> 1)
4. Sigmoid

Let:

1. $c$ = classical logits
2. $q$ = quantum logits
3. $\alpha \in [0,1]$ = adaptive gate output

Final logits are:

$$
\\text{logits} = \alpha q + (1-\alpha)c
$$

Interpretation:

1. High $\alpha$ means stronger quantum influence
2. Low $\alpha$ means stronger classical influence

## 6. Apple Silicon (M1) compatibility behavior

The project is intentionally patched for M1 stability:

1. CNN and dense layers run on MPS when available
2. Quantum simulator runs on CPU tensors
3. For MPS execution, the quantum path is detached from the MPS autograd graph to avoid float64 backend incompatibility

Why this is needed:

1. PennyLane simulator internally uses operations that may involve float64 behavior
2. MPS backend does not support float64 in autograd paths
3. Without this workaround, training can fail with dtype/CUDA-style compatibility errors

Practical effect:

1. Model still trains and predicts correctly on M1
2. Quantum branch contributes to outputs
3. Quantum parameters are not fully backpropagated through MPS path in this compatibility mode

## 7. Training pipeline deep dive

Defined in train.py.

### 7.1 Device selection

Priority order:

1. mps
2. cuda
3. cpu

### 7.2 Data transforms

Train transforms:

1. Grayscale -> 3-channel
2. Resize to 224x224
3. RandomHorizontalFlip
4. RandomRotation(8)
5. ToTensor
6. ImageNet normalization

Validation/test transforms:

1. Grayscale -> 3-channel
2. Resize 224x224
3. ToTensor
4. ImageNet normalization

### 7.3 Dataloaders

1. ImageFolder for train/val/test
2. Batch size from CLI
3. num_workers=2
4. Train loader shuffles, val/test do not

### 7.4 Optimization strategy

1. Loss: CrossEntropyLoss
2. Optimizer: AdamW
3. Initial phase: MobileNet features frozen
4. At unfreeze_epoch: backbone unfrozen and LR reduced (lr * 0.3)

### 7.5 Metrics tracked per epoch

1. loss
2. acc
3. adaptive alpha mean

### 7.6 Checkpointing

Best model (by validation accuracy) is saved to:

1. checkpoints/best_model.pt

Checkpoint payload contains:

1. model_state
2. config (n_qubits, q_layers)
3. best_val_acc

### 7.7 Final test evaluation

After training, best checkpoint is reloaded and tested on the test set.

## 8. Inference service and communication flow

Backend defined in app/main.py and app/inference.py.

### 8.1 Backend routes

1. GET / : serves frontend/index.html
2. GET /health : returns status and model readiness
3. POST /predict : accepts uploaded image and returns prediction JSON

### 8.2 Request/response contract for /predict

Input:

1. multipart/form-data with file field named file

Validation:

1. content type must start with image/
2. image must be decodable by PIL

Output JSON:

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

Failure behavior:

1. 400 for invalid file/type
2. 503 if checkpoint is not found (model not trained yet)

## 9. Frontend behavior and API communication

Frontend files:

1. frontend/index.html
2. frontend/styles.css
3. frontend/app.js

Flow:

1. User chooses an image from file input
2. JS previews image in browser via object URL
3. On submit, JS creates FormData and sends POST /predict
4. UI displays class prediction, confidence, class probabilities, adaptive quantum contribution

## 10. Setup and run instructions (copy-paste)

From project root:

```bash
cd "/Users/irfanmohammed/Desktop/Final YEAR PROJ"
```

Create venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train (full):

```bash
"/Users/irfanmohammed/Desktop/Final YEAR PROJ/.venv/bin/python" \
"/Users/irfanmohammed/Desktop/Final YEAR PROJ/train.py" \
--epochs 8 --batch_size 8 --n_qubits 4 --q_layers 2
```

Train (quick smoke test):

```bash
"/Users/irfanmohammed/Desktop/Final YEAR PROJ/.venv/bin/python" \
"/Users/irfanmohammed/Desktop/Final YEAR PROJ/train.py" \
--epochs 1 --batch_size 4 --n_qubits 4 --q_layers 2
```

Run web app:

```bash
cd "/Users/irfanmohammed/Desktop/Final YEAR PROJ"
"/Users/irfanmohammed/Desktop/Final YEAR PROJ/.venv/bin/python" -m uvicorn app.main:app --reload
```

Open:

1. http://127.0.0.1:8000

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

## 11. CLI argument reference for training

train.py supports:

1. --train_dir (default: chest_xray/train)
2. --val_dir (default: chest_xray/val)
3. --test_dir (default: chest_xray/test)
4. --epochs (default: 8)
5. --batch_size (default: 8)
6. --lr (default: 2e-4)
7. --n_qubits (default: 4)
8. --q_layers (default: 2)
9. --unfreeze_epoch (default: 4)

## 12. Expected logs and outputs

During training, expect lines like:

```text
Using device: mps
Epoch 1/8 | train_loss=... train_acc=... val_loss=... val_acc=... adaptive_alpha=...
Saved new best model with val_acc=...
Evaluating best model on test set...
Test metrics | loss=... acc=... adaptive_alpha=...
```

Artifacts produced:

1. checkpoints/best_model.pt

## 13. Common issues and fixes

### Issue: command fails because of spaces in folder name

Use quoted absolute paths as shown in section 10.

### Issue: /predict returns "Model checkpoint not found"

Train first so checkpoints/best_model.pt exists.

### Issue: slower training on M1

Reduce complexity:

1. lower batch size (4 or 8)
2. fewer epochs for iteration
3. keep n_qubits small (2 to 4)
4. keep q_layers small (1 to 2)

### Issue: high validation variance

1. run multiple seeds
2. try lower learning rate
3. increase epochs moderately
4. inspect class imbalance and sample quality

## 14. Current limitations

1. No Grad-CAM or saliency visualization yet
2. No experiment tracker integration (TensorBoard/WandB)
3. No calibration metrics (AUC, sensitivity, specificity) yet
4. Quantum branch uses simulator, not real quantum hardware

## 15. Suggested next upgrades

1. Add confusion matrix, ROC-AUC, precision/recall/F1 reporting
2. Add Grad-CAM heatmaps in the frontend
3. Add model/version metadata endpoint
4. Add Docker setup for portable deployment
5. Add test suite for API endpoints and inference pipeline

## 16. Academic/reporting note

For a final year report, present:

1. Baseline classical CNN results
2. Hybrid model results
3. Ablation across n_qubits, q_layers, and unfreeze strategies
4. Discussion of M1 compatibility trade-offs and simulator overhead

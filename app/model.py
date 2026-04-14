import math

import pennylane as qml
import torch
import torch.nn as nn
import torchvision.models as models


class QuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.Rot(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PennyLane default simulator is most stable with CPU torch tensors.
        x_cpu = x.to("cpu")
        w_cpu = self.weights.to("cpu")
        outputs = [torch.stack(self.circuit(sample, w_cpu)) for sample in x_cpu]
        return torch.stack(outputs)


class AdaptiveHybridQCNN(nn.Module):
    def __init__(self, n_qubits: int = 4, q_layers: int = 2, freeze_backbone: bool = True):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        cnn_feat_dim = 576
        self.reduce = nn.Sequential(
            nn.Linear(cnn_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_qubits),
        )

        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=q_layers)

        self.classical_head = nn.Linear(n_qubits, 2)
        self.quantum_head = nn.Linear(n_qubits, 2)

        self.adaptive_gate = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        feat = self.features(x)
        feat = self.avgpool(feat).flatten(1)
        latent = self.reduce(feat)

        bounded_latent = torch.tanh(latent) * math.pi

        # On MPS, keep quantum simulation out of the backward graph to avoid
        # float64 autograd incompatibilities in Apple's backend.
        if latent.device.type == "mps":
            q_in = bounded_latent.detach()
        else:
            q_in = bounded_latent

        q_out = self.quantum(q_in).to(device=latent.device, dtype=latent.dtype)

        if latent.device.type == "mps":
            q_out = q_out.detach()

        c_logits = self.classical_head(latent)
        q_logits = self.quantum_head(q_out)

        alpha = self.adaptive_gate(latent)
        logits = alpha * q_logits + (1.0 - alpha) * c_logits

        return logits, alpha.mean().detach()

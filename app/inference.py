from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from app.model import AdaptiveHybridQCNN


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class Predictor:
    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pt"):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self._get_device()
        self.model = None

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._load_model()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load_model(self):
        if not self.checkpoint_path.exists():
            self.model = None
            return

        payload = torch.load(self.checkpoint_path, map_location=self.device)
        config = payload.get("config", {})
        self.model = AdaptiveHybridQCNN(
            n_qubits=config.get("n_qubits", 4),
            q_layers=config.get("q_layers", 2),
            freeze_backbone=False,
        ).to(self.device)
        self.model.quantum.to(torch.device("cpu"))
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def is_ready(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        if self.model is None:
            raise RuntimeError("Model checkpoint not found. Train the model first.")

        x = self.transform(image).unsqueeze(0).to(self.device)
        logits, adaptive_weight = self.model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        class_probs = {CLASS_NAMES[i]: float(probs[i].item()) for i in range(2)}
        class_probs["adaptive_quantum_weight"] = float(adaptive_weight.item())

        return CLASS_NAMES[pred_idx], confidence, class_probs

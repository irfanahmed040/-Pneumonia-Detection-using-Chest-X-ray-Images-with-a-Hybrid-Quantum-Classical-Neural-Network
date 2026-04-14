import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from app.model import AdaptiveHybridQCNN


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loaders(train_dir: str, val_dir: str, test_dir: str, batch_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    correct = 0
    total = 0
    adaptive_alpha = 0.0

    ctx = torch.enable_grad() if train_mode else torch.inference_mode()
    with ctx:
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)

            logits, alpha = model(x)
            loss = criterion(logits, y)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(x.size(0))
            adaptive_alpha += float(alpha.item()) * x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "alpha": adaptive_alpha / max(total, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="chest_xray/train")
    parser.add_argument("--val_dir", default="chest_xray/val")
    parser.add_argument("--test_dir", default="chest_xray/test")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--q_layers", type=int, default=2)
    parser.add_argument("--unfreeze_epoch", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_loaders(
        args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    model = AdaptiveHybridQCNN(
        n_qubits=args.n_qubits,
        q_layers=args.q_layers,
        freeze_backbone=True,
    ).to(device)
    model.quantum.to(torch.device("cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val_acc = 0.0
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=args.lr * 0.3)
            model.quantum.to(torch.device("cpu"))
            print("Backbone unfrozen with lower LR.")

        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"adaptive_alpha={val_metrics['alpha']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "n_qubits": args.n_qubits,
                        "q_layers": args.q_layers,
                    },
                    "best_val_acc": best_val_acc,
                },
                ckpt_dir / "best_model.pt",
            )
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

    print("Evaluating best model on test set...")
    payload = torch.load(ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(payload["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None)
    print(
        f"Test metrics | loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['acc']:.4f} adaptive_alpha={test_metrics['alpha']:.4f}"
    )


if __name__ == "__main__":
    main()

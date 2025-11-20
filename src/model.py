from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class TitanicNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: Iterable[int] = (256, 128, 64, 32),
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        
        # First layer with more capacity
        layers.extend([
            nn.Linear(last_dim, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout in first layer
        ])
        last_dim = hidden_units[0]
        
        # Middle layers
        for units in hidden_units[1:]:
            layers.extend(
                [
                    nn.Linear(last_dim, units),
                    nn.BatchNorm1d(units),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            last_dim = units
        
        # Output layer
        layers.append(nn.Linear(last_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Handle batch size 1 for BatchNorm1d (requires batch_size > 1 during training)
        if self.training and x.size(0) == 1:
            # Temporarily set to eval mode for single-sample batches
            self.eval()
            output = self.network(x)
            self.train()
            return output
        return self.network(x)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
        }


def _binary_accuracy(logits: Tensor, targets: Tensor) -> float:
    preds = torch.sigmoid(logits).ge(0.5).float()
    correct = (preds == targets).float().mean().item()
    return correct


def train_model(
    model: TitanicNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_dir: Path,
    device: torch.device,
    epochs: int = 200,
    lr: float = 5e-4,
    patience: int = 20,
    min_delta: float = 0.0,
) -> TrainingHistory:
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()
    history = TrainingHistory([], [], [], [])
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
            train_acc += _binary_accuracy(outputs.detach(), labels) * features.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * features.size(0)
                val_acc += _binary_accuracy(outputs, labels) * features.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        # Track best validation accuracy instead of loss
        # Only reset patience if improvement is significant (min_delta)
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        scheduler.step(val_acc)  # Step scheduler based on validation accuracy

        print(
            f"Epoch {epoch:03d} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} "
            f"(best: {best_val_acc:.4f})"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered. Best validation accuracy: {best_val_acc:.4f}")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set to eval mode before returning (important for BatchNorm/Dropout)
    return history


def export_full_model(model: TitanicNet, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), export_path)

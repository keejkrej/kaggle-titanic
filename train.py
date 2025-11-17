from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import PreprocessedData, load_training_data, transform_test_data
from src.model import TitanicNet, export_full_model, train_model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DNN classifier for the Kaggle Titanic challenge.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train.csv and test.csv.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for saving trained models.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"), help="Directory for saving preprocessing artifacts.")
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"), help="Directory for submission files.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="Device to train on: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--generate-submission", action="store_true", help="Generate a Kaggle-ready submission after training.")
    return parser.parse_args()


def _save_history(history: dict[str, list[float]], artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifact_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)
    return history_path


def _make_dataloaders(
    data: PreprocessedData,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    x_train_tensor = torch.from_numpy(data.x_train).float()
    y_train_tensor = torch.from_numpy(data.y_train).float().unsqueeze(1)
    x_val_tensor = torch.from_numpy(data.x_val).float()
    y_val_tensor = torch.from_numpy(data.y_val).float().unsqueeze(1)
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _generate_submission(
    model: TitanicNet,
    preprocessor: PreprocessedData,
    data_dir: Path,
    submissions_dir: Path,
    device: torch.device,
) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    passenger_ids, test_features = transform_test_data(data_dir, preprocessor.preprocessor)
    with torch.no_grad():
        feature_tensor = torch.from_numpy(test_features).float().to(device)
        logits = model(feature_tensor)
        predictions = torch.sigmoid(logits).cpu().numpy().ravel()
    predicted_labels = (predictions >= 0.5).astype(int)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    submission_path = submissions_dir / f"submission_{timestamp}.csv"
    submission_df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predicted_labels})
    submission_df.to_csv(submission_path, index=False)
    return submission_path


def main() -> None:
    args = parse_args()
    _set_seeds(args.random_state)
    device = _get_device(args.device)
    print(f"Using device: {device}")

    data = load_training_data(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_loader, val_loader = _make_dataloaders(data, args.batch_size)

    model = TitanicNet(input_dim=data.x_train.shape[1])
    model.to(device)
    history_obj = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=args.model_dir,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )
    history = history_obj.as_dict()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_path = _save_history(history, artifact_dir)
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    joblib.dump(data.preprocessor, preprocessor_path)

    final_model_path = args.model_dir / "titanic_dnn.pt"
    export_full_model(model, final_model_path)

    val_loss = history["val_loss"][-1]
    val_acc = history["val_accuracy"][-1]
    print(f"Validation loss: {val_loss:.4f} - Validation accuracy: {val_acc:.4f}")
    print(f"Saved model to {final_model_path}")
    print(f"Saved preprocessing pipeline to {preprocessor_path}")
    print(f"Training history stored at {history_path}")

    if args.generate_submission:
        submission_path = _generate_submission(model, data, args.data_dir, args.submissions_dir, device)
        print(f"Submission file created at {submission_path}")


if __name__ == "__main__":
    main()

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
    parser.add_argument("--test-size", type=float, default=0.1)  # Even smaller validation set = more training data
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
    # Drop last batch if it's size 1 to avoid BatchNorm issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _generate_submission(
    model,
    preprocessor: PreprocessedData,
    data_dir: Path,
    submissions_dir: Path,
    device: torch.device,
) -> Path:
    """Generate submission file. Model can be TitanicNet or EnsembleModel."""
    from src.ensemble import EnsembleModel
    
    submissions_dir.mkdir(parents=True, exist_ok=True)
    passenger_ids, test_features = transform_test_data(data_dir, preprocessor.preprocessor)
    
    if isinstance(model, EnsembleModel):
        predicted_labels = model.predict(test_features)
    else:
        model.eval()  # Set to eval mode for inference (important for BatchNorm/Dropout)
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
    device = _get_device(args.device)
    print(f"Using device: {device}")

    data = load_training_data(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    
    # Train ensemble of models with different random seeds
    n_models = 30  # Increased to 30 models for better ensemble
    ensemble_models = []
    all_histories = []
    
    print(f"\nTraining ensemble of {n_models} models...")
    # More diverse architectures for maximum diversity
    architectures = [
        (512, 256, 128, 64, 32),
        (256, 128, 64, 32),
        (1024, 512, 256, 128),
        (128, 64, 32, 16),
        (256, 128, 64),
        (512, 256, 128),
        (256, 256, 128, 64),
        (512, 512, 256, 128),
        (128, 128, 64, 32),
        (256, 256, 128),
        (1024, 256, 128, 64),
        (512, 128, 64, 32),
        (256, 256, 256, 128),
        (128, 256, 128, 64),
        (512, 512, 512, 256),
    ] * 2  # Repeat to get 30 models
    
    for i in range(n_models):
        print(f"\n--- Training model {i+1}/{n_models} ---")
        seed = args.random_state + i * 1000
        _set_seeds(seed)
        
        train_loader, val_loader = _make_dataloaders(data, args.batch_size)
        # Use different architectures for diversity
        hidden_units = architectures[i % len(architectures)]
        model = TitanicNet(input_dim=data.x_train.shape[1], hidden_units=hidden_units)
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
        
        # Save individual model
        model_path = args.model_dir / f"titanic_dnn_ensemble_{i+1}.pt"
        export_full_model(model, model_path)
        
        # Save architecture info
        arch_info_path = args.model_dir / f"titanic_dnn_ensemble_{i+1}_arch.json"
        arch_info = {"hidden_units": list(hidden_units)}
        with arch_info_path.open("w") as f:
            json.dump(arch_info, f)
        
        ensemble_models.append({
            "model_path": model_path,
            "arch_path": arch_info_path,
            "hidden_units": hidden_units,
        })
        all_histories.append(history_obj.as_dict())
        
        best_val_acc = max(history_obj.val_accuracy)
        print(f"Model {i+1} best validation accuracy: {best_val_acc:.4f}")
    
    # Evaluate ensemble on validation set
    print(f"\n--- Evaluating ensemble ---")
    from src.ensemble import EnsembleModel
    ensemble = EnsembleModel.load_ensemble_from_info(
        ensemble_models, data.x_train.shape[1], device
    )
    
    # Calculate ensemble accuracy on validation set (with and without weights)
    val_predictions = ensemble.predict(data.x_val)
    ensemble_val_acc = (val_predictions == data.y_val).mean()
    
    # Try weighted ensemble based on individual model performance
    individual_accs = np.array([float(max(h["val_accuracy"])) for h in all_histories])
    # Use accuracy as weights (higher accuracy = higher weight)
    weights = individual_accs ** 2  # Square to emphasize better models
    weighted_val_predictions = ensemble.predict(data.x_val, weights=weights)
    weighted_ensemble_val_acc = (weighted_val_predictions == data.y_val).mean()
    
    print(f"Ensemble validation accuracy (equal weights): {ensemble_val_acc:.4f}")
    print(f"Ensemble validation accuracy (weighted): {weighted_ensemble_val_acc:.4f}")
    
    # Use the better performing ensemble
    if weighted_ensemble_val_acc > ensemble_val_acc:
        ensemble_val_acc = weighted_ensemble_val_acc
        ensemble_weights = weights.tolist()
        print("Using weighted ensemble")
    else:
        ensemble_weights = None
        print("Using equal-weight ensemble")
    
    # Save ensemble model paths
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the first model as the main model for backward compatibility
    final_model_path = args.model_dir / "titanic_dnn.pt"
    export_full_model(ensemble.models[0], final_model_path)
    
    # Save ensemble info
    ensemble_info = {
        "model_info": [
            {
                "model_path": str(info["model_path"]),
                "arch_path": str(info["arch_path"]),
                "hidden_units": list(info["hidden_units"]),
            }
            for info in ensemble_models
        ],
        "ensemble_val_accuracy": float(ensemble_val_acc),
        "individual_accuracies": [float(max(h["val_accuracy"])) for h in all_histories],
        "ensemble_weights": ensemble_weights,
    }
    ensemble_info_path = artifact_dir / "ensemble_info.json"
    with ensemble_info_path.open("w") as f:
        json.dump(ensemble_info, f, indent=2)
    
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    joblib.dump(data.preprocessor, preprocessor_path)
    
    # Save average history (handle different lengths)
    max_len = max(len(h["train_loss"]) for h in all_histories)
    avg_history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    for i in range(max_len):
        train_losses = [h["train_loss"][i] for h in all_histories if i < len(h["train_loss"])]
        train_accs = [h["train_accuracy"][i] for h in all_histories if i < len(h["train_accuracy"])]
        val_losses = [h["val_loss"][i] for h in all_histories if i < len(h["val_loss"])]
        val_accs = [h["val_accuracy"][i] for h in all_histories if i < len(h["val_accuracy"])]
        
        if train_losses:
            avg_history["train_loss"].append(np.mean(train_losses))
            avg_history["train_accuracy"].append(np.mean(train_accs))
            avg_history["val_loss"].append(np.mean(val_losses))
            avg_history["val_accuracy"].append(np.mean(val_accs))
    
    history_path = _save_history(avg_history, artifact_dir)
    
    print(f"\nBest individual validation accuracy: {max([max(h['val_accuracy']) for h in all_histories]):.4f}")
    print(f"Ensemble validation accuracy: {ensemble_val_acc:.4f}")
    
    # Stage 2: Retrain best models on FULL dataset for final ensemble
    print(f"\n--- Stage 2: Retraining best models on FULL dataset ---")
    from src.data_utils import load_full_training_data
    
    # Select top N models based on validation accuracy
    model_accuracies = [(i, max(h["val_accuracy"])) for i, h in enumerate(all_histories)]
    model_accuracies.sort(key=lambda x: x[1], reverse=True)
    top_n = min(20, n_models)  # Retrain top 20 models on full dataset
    
    print(f"Retraining top {top_n} models on full dataset...")
    full_features, full_labels, full_preprocessor, feature_names = load_full_training_data(
        args.data_dir, args.random_state
    )
    
    # Create full dataset loader
    full_x_tensor = torch.from_numpy(full_features).float()
    full_y_tensor = torch.from_numpy(full_labels).float().unsqueeze(1)
    full_dataset = TensorDataset(full_x_tensor, full_y_tensor)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # Create a small holdout for monitoring (but train on everything)
    val_loader_for_monitoring = _make_dataloaders(data, args.batch_size)[1]
    
    final_ensemble_models = []
    for rank, (model_idx, acc) in enumerate(model_accuracies[:top_n]):
        print(f"\n--- Retraining model {rank+1}/{top_n} (was model {model_idx+1}, acc: {acc:.4f}) ---")
        seed = args.random_state + model_idx * 1000
        _set_seeds(seed)
        
        hidden_units = ensemble_models[model_idx]["hidden_units"]
        model = TitanicNet(input_dim=full_features.shape[1], hidden_units=hidden_units)
        model.to(device)
        
        # Train on full dataset with more epochs and patience
        history_obj = train_model(
            model=model,
            train_loader=full_loader,
            val_loader=val_loader_for_monitoring,  # Use original val set for monitoring
            model_dir=args.model_dir,
            device=device,
            epochs=args.epochs * 2,  # More epochs for full dataset
            lr=args.lr * 0.7,  # Lower LR for more stable training on full dataset
            patience=30,  # More patience for full dataset training
            min_delta=0.0001,  # Require small improvement to reset patience
        )
        
        # Save final model
        final_model_path = args.model_dir / f"titanic_dnn_final_{rank+1}.pt"
        export_full_model(model, final_model_path)
        
        final_arch_path = args.model_dir / f"titanic_dnn_final_{rank+1}_arch.json"
        arch_info = {"hidden_units": list(hidden_units)}
        with final_arch_path.open("w") as f:
            json.dump(arch_info, f)
        
        final_ensemble_models.append({
            "model_path": final_model_path,
            "arch_path": final_arch_path,
            "hidden_units": hidden_units,
        })
    
    # Create final ensemble from models trained on full data
    print(f"\n--- Creating final ensemble from full-dataset models ---")
    final_ensemble = EnsembleModel.load_ensemble_from_info(
        final_ensemble_models, full_features.shape[1], device
    )
    
    # Evaluate on validation set
    final_val_predictions = final_ensemble.predict(data.x_val)
    final_ensemble_val_acc = (final_val_predictions == data.y_val).mean()
    
    # Try weighted ensemble
    final_individual_accs = np.array([acc for _, acc in model_accuracies[:top_n]])
    final_weights = final_individual_accs ** 3  # Cube to emphasize best models even more
    final_weighted_predictions = final_ensemble.predict(data.x_val, weights=final_weights)
    final_weighted_acc = (final_weighted_predictions == data.y_val).mean()
    
    print(f"Final ensemble validation accuracy (equal weights): {final_ensemble_val_acc:.4f}")
    print(f"Final ensemble validation accuracy (weighted): {final_weighted_acc:.4f}")
    
    # Use the better one
    if final_weighted_acc > final_ensemble_val_acc:
        final_ensemble_val_acc = final_weighted_acc
        final_ensemble_weights = final_weights.tolist()
        print("Using weighted final ensemble")
    else:
        final_ensemble_weights = None
        print("Using equal-weight final ensemble")
    
    # Update ensemble info with final models
    ensemble_info["final_model_info"] = [
        {
            "model_path": str(info["model_path"]),
            "arch_path": str(info["arch_path"]),
            "hidden_units": list(info["hidden_units"]),
        }
        for info in final_ensemble_models
    ]
    ensemble_info["final_ensemble_val_accuracy"] = float(final_ensemble_val_acc)
    ensemble_info["final_ensemble_weights"] = final_ensemble_weights
    
    # Save updated preprocessor (trained on full data)
    joblib.dump(full_preprocessor, preprocessor_path)
    
    with ensemble_info_path.open("w") as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Best individual validation accuracy: {max([max(h['val_accuracy']) for h in all_histories]):.4f}")
    print(f"Initial ensemble validation accuracy: {ensemble_val_acc:.4f}")
    print(f"Final ensemble validation accuracy (full dataset): {final_ensemble_val_acc:.4f}")
    print(f"{'='*60}")
    print(f"Saved ensemble models to {args.model_dir}")
    print(f"Saved preprocessing pipeline to {preprocessor_path}")
    print(f"Training history stored at {history_path}")
    print(f"Ensemble info stored at {ensemble_info_path}")

    if args.generate_submission:
        # Use final ensemble for submission
        submission_path = _generate_submission(
            final_ensemble, 
            type('obj', (object,), {'preprocessor': full_preprocessor})(), 
            args.data_dir, 
            args.submissions_dir, 
            device
        )
        print(f"Submission file created at {submission_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch

from src.data_utils import transform_test_data
from src.model import TitanicNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("models/titanic_dnn.pt"))
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=Path("artifacts/preprocessor.joblib"),
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=Path("submissions/submission.csv"),
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference.")
    return parser.parse_args()


def _get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = parse_args()
    device = _get_device(args.device)

    if not args.preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {args.preprocessor_path}. Train the model with train.py first."
        )

    preprocessor = joblib.load(args.preprocessor_path)
    passenger_ids, test_features = transform_test_data(args.data_dir, preprocessor)
    
    # Check if ensemble info exists
    ensemble_info_path = args.preprocessor_path.parent / "ensemble_info.json"
    if ensemble_info_path.exists():
        # Use ensemble
        from src.ensemble import EnsembleModel
        
        # Handle both old and new format
        import json
        with ensemble_info_path.open() as f:
            ensemble_info = json.load(f)
        
        if "model_info" in ensemble_info:
            # Prefer final models trained on full dataset if available
            if "final_model_info" in ensemble_info:
                # Use final ensemble
                final_ensemble = EnsembleModel.load_ensemble_from_info(
                    ensemble_info["final_model_info"], test_features.shape[1], device
                )
                weights = ensemble_info.get("final_ensemble_weights")
                if weights:
                    import numpy as np
                    weights = np.array(weights)
                    predicted_labels = final_ensemble.predict(test_features, weights=weights)
                    print(f"Using weighted final ensemble of {len(final_ensemble.models)} models")
                else:
                    predicted_labels = final_ensemble.predict(test_features)
                    print(f"Using final ensemble of {len(final_ensemble.models)} models")
            else:
                # Use initial ensemble
                ensemble = EnsembleModel.load_ensemble_from_info(
                    ensemble_info["model_info"], test_features.shape[1], device
                )
                weights = ensemble_info.get("ensemble_weights")
                if weights:
                    import numpy as np
                    weights = np.array(weights)
                    predicted_labels = ensemble.predict(test_features, weights=weights)
                    print(f"Using weighted ensemble of {len(ensemble.models)} models")
                else:
                    predicted_labels = ensemble.predict(test_features)
                    print(f"Using ensemble of {len(ensemble.models)} models")
        else:
            # Old format (backward compatibility)
            model_paths = [Path(p) for p in ensemble_info["model_paths"]]
            ensemble = EnsembleModel.load_ensemble(
                model_paths, test_features.shape[1], device
            )
            predicted_labels = ensemble.predict(test_features)
            print(f"Using ensemble of {len(ensemble.models)} models")
    else:
        # Use single model
        if not args.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {args.model_path}. Train the model with train.py first."
            )
        model = TitanicNet(input_dim=test_features.shape[1])
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        with torch.no_grad():
            feature_tensor = torch.from_numpy(test_features).float().to(device)
            logits = model(feature_tensor)
            predictions = torch.sigmoid(logits).cpu().numpy().ravel()
        predicted_labels = (predictions >= 0.5).astype(int)

    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predicted_labels})
    submission_df.to_csv(args.submission_path, index=False)
    print(f"Saved submission to {args.submission_path}")


if __name__ == "__main__":
    main()

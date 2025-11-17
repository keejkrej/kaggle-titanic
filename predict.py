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

    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Train the model with train.py first."
        )
    if not args.preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {args.preprocessor_path}. Train the model with train.py first."
        )

    preprocessor = joblib.load(args.preprocessor_path)
    passenger_ids, test_features = transform_test_data(args.data_dir, preprocessor)
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

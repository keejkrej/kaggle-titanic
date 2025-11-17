from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

from src.data_utils import transform_test_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kaggle submission predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=Path("models/titanic_dnn.keras"))
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {args.model_path}. Train the model with train.py first."
        )
    if not args.preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {args.preprocessor_path}. Train the model with train.py first."
        )
    model = tf.keras.models.load_model(args.model_path)
    preprocessor = joblib.load(args.preprocessor_path)
    passenger_ids, test_features = transform_test_data(args.data_dir, preprocessor)
    predictions = (model.predict(test_features, verbose=0).ravel() >= 0.5).astype(int)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    submission_df.to_csv(args.submission_path, index=False)
    print(f"Saved submission to {args.submission_path}")


if __name__ == "__main__":
    main()

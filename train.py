from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_utils import PreprocessedData, load_training_data, transform_test_data
from src.model import build_model, export_full_model, train_model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DNN classifier for the Kaggle Titanic challenge.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing train.csv and test.csv.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory for saving trained models.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"), help="Directory for saving preprocessing artifacts.")
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"), help="Directory for submission files.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--generate-submission", action="store_true", help="Generate a Kaggle-ready submission after training.")
    return parser.parse_args()


def _save_history(history: tf.keras.callbacks.History, artifact_dir: Path) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifact_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history.history, f, indent=2)
    return history_path


def _generate_submission(
    model: tf.keras.Model,
    preprocessor: PreprocessedData,
    data_dir: Path,
    submissions_dir: Path,
) -> Path:
    submissions_dir.mkdir(parents=True, exist_ok=True)
    passenger_ids, test_features = transform_test_data(data_dir, preprocessor.preprocessor)
    predictions = (model.predict(test_features, verbose=0).ravel() >= 0.5).astype(int)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    submission_path = submissions_dir / f"submission_{timestamp}.csv"
    submission_df = pd.DataFrame({"PassengerId": passenger_ids, "Survived": predictions})
    submission_df.to_csv(submission_path, index=False)
    return submission_path


def main() -> None:
    args = parse_args()
    _set_seeds(args.random_state)
    data = load_training_data(
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    model = build_model(input_dim=data.x_train.shape[1])
    history = train_model(
        model=model,
        x_train=data.x_train,
        y_train=data.y_train,
        x_val=data.x_val,
        y_val=data.y_val,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history_path = _save_history(history, artifact_dir)
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    joblib.dump(data.preprocessor, preprocessor_path)

    eval_loss, eval_accuracy = model.evaluate(data.x_val, data.y_val, verbose=0)
    print(f"Validation loss: {eval_loss:.4f} - Validation accuracy: {eval_accuracy:.4f}")

    final_model_path = args.model_dir / "titanic_dnn.keras"
    export_full_model(model, final_model_path)
    print(f"Saved model to {final_model_path}")
    print(f"Saved preprocessing pipeline to {preprocessor_path}")
    print(f"Training history stored at {history_path}")

    if args.generate_submission:
        submission_path = _generate_submission(model, data, args.data_dir, args.submissions_dir)
        print(f"Submission file created at {submission_path}")


if __name__ == "__main__":
    main()

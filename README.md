# Kaggle Titanic - Deep Neural Network Baseline

This repository provides a reproducible workflow for training a deep neural network on the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/overview). The pipeline handles feature engineering, preprocessing, model training, and submission generation.

## Project structure

```
.
├── artifacts/              # Saved preprocessing pipeline + training history
├── data/                   # Place Kaggle train.csv and test.csv here
├── models/                 # Trained PyTorch models
├── src/
│   ├── data_utils.py       # Feature engineering + preprocessing helpers
│   └── model.py            # Model architecture + training utilities
├── submissions/            # Kaggle-ready CSV files
├── predict.py              # Generate submission from saved artifacts
├── train.py                # Main training script
└── requirements.txt        # Python dependencies
```

## Getting the data

1. Download the competition files (`train.csv`, `test.csv`) from Kaggle. This requires logging in and accepting the competition rules.
2. Place both files under the `data/` directory in this project:

```
/Users/jack/workspace/kaggle-titanic
└── data
    ├── train.csv
    └── test.csv
```

If you have the Kaggle CLI configured locally, run the following from the project root:

```bash
kaggle competitions download -c titanic -p data
unzip data/titanic.zip -d data
```

## Setup

Create and activate a Python environment (3.10+ recommended), then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python train.py --generate-submission
```

Key arguments:

- `--data-dir`: where the CSV files live (`data` by default)
- `--epochs`: maximum training epochs (default 200 with early stopping)
- `--batch-size`: batch size for PyTorch DataLoaders (default 32)
- `--device`: `cpu`, `cuda`, or `auto` (default `auto`)
- `--test-size`: validation split size (default 0.2)
- `--generate-submission`: immediately create a submission after training

Artifacts saved:

- `models/titanic_dnn.pt`: final PyTorch weights (state dict)
- `models/best_model.pt`: checkpoint with the best validation loss
- `artifacts/preprocessor.joblib`: fitted `ColumnTransformer`
- `artifacts/training_history.json`: loss/accuracy per epoch
- `submissions/submission_*.csv`: Kaggle submission (only when `--generate-submission` is provided)

## Generating submissions later

After training, regenerate predictions at any time without retraining:

```bash
python predict.py \
  --data-dir data \
  --model-path models/titanic_dnn.pt \
  --preprocessor-path artifacts/preprocessor.joblib \
  --submission-path submissions/submission.csv \
  --device auto
```

## Notes

- The deep neural network uses engineered features (family size, cabin deck, ticket group size, passenger titles) and a preprocessing pipeline that is persisted for inference.
- Training and inference are implemented with PyTorch; no TensorFlow dependency is required.
- Because the Kaggle dataset requires authentication, this repository does **not** download the data automatically. Ensure the CSV files are in `data/` before running the scripts.

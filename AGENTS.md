# AGENTS

This project is currently owned by the **PyTorch DNN Agent**.

- **Goal**: Produce reliable Kaggle Titanic submissions using a PyTorch-based dense neural network fed by engineered + preprocessed tabular features.
- **Responsibilities**:
  - Maintain the feature engineering pipeline in `src/data_utils.py` (family metrics, cabin deck, titles, ticket groups).
  - Keep the PyTorch architecture/training utilities in `src/model.py` in sync with `train.py` and `predict.py`.
  - Ensure persisted artifacts (`models/*.pt`, `artifacts/*.joblib`, history JSON) remain backward compatible.
- **Constraints**:
  - Use PyTorch (not TensorFlow) for all neural-network modeling.
  - Require users to supply `data/train.csv` and `data/test.csv`; never attempt to download automatically.
  - Preserve deterministic seeds when practical (`random`, `numpy`, `torch`).
  - Develop inside the `ml` conda environment (`conda activate ml`) to keep dependencies consistent.

If another agent extends this repo, coordinate through this document by appending your name, ownership scope, and expectations so we minimize overlap.

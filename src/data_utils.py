from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "SibSp",
    "Parch",
    "FamilySize",
    "TicketGroupSize",
]
BOOLEAN_FEATURES = ["IsAlone"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass", "Title", "CabinDeck"]
TARGET_COLUMN = "Survived"


@dataclass
class PreprocessedData:
    x_train: np.ndarray
    x_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]


def _extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    parts = name.split(", ")
    if len(parts) < 2:
        return "Unknown"
    title_section = parts[1]
    title = title_section.split(".")[0]
    title = title.strip()
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Noble",
        "Countess": "Noble",
        "Dona": "Noble",
        "Dr": "Officer",
        "Major": "Officer",
        "Col": "Officer",
        "Capt": "Officer",
        "Sir": "Noble",
        "Don": "Noble",
        "Jonkheer": "Noble",
        "Rev": "Clergy",
    }
    return title_map.get(title, title)


def _extract_cabin_deck(cabin_value: str) -> str:
    if isinstance(cabin_value, str) and cabin_value:
        return cabin_value[0]
    return "Unknown"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    engineered["FamilySize"] = engineered["SibSp"].fillna(0) + engineered["Parch"].fillna(0) + 1
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    engineered["Title"] = engineered["Name"].apply(_extract_title)
    engineered["CabinDeck"] = engineered["Cabin"].apply(_extract_cabin_deck)
    engineered["TicketGroupSize"] = (
        engineered.groupby("Ticket")["Ticket"].transform("count").clip(lower=1)
    )
    return engineered


def _build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    boolean_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("bool", boolean_transformer, BOOLEAN_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )
    return preprocessor


def _validate_data_path(data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Expected data file at {data_path}. "
            "Please download the Kaggle Titanic dataset and place the CSV files under the data/ directory."
        )


def load_training_data(
    data_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    train_path = data_dir / "train.csv"
    _validate_data_path(train_path)
    raw_df = pd.read_csv(train_path)
    engineered_df = _engineer_features(raw_df)
    y = engineered_df[TARGET_COLUMN].to_numpy()
    feature_df = engineered_df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = _build_preprocessor()
    features = preprocessor.fit_transform(feature_df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return PreprocessedData(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )


def transform_test_data(
    data_dir: Path,
    preprocessor: ColumnTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    test_path = data_dir / "test.csv"
    _validate_data_path(test_path)
    raw_df = pd.read_csv(test_path)
    passenger_ids = raw_df["PassengerId"].to_numpy()
    engineered_df = _engineer_features(raw_df)
    feature_df = engineered_df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    test_features = preprocessor.transform(feature_df)
    return passenger_ids, test_features

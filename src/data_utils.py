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
    "FarePerPerson",
    "AgeGroup",
    "FareGroup",
    # Interaction features
    "Sex_Pclass",
    "Age_Pclass",
    "Fare_Pclass",
    "FamilySize_Pclass",
    "Age_Sex",
    "Fare_Sex",
    "Age_FamilySize",
    "FarePerPersonSq",
    "CabinDeckNumeric",
]
BOOLEAN_FEATURES = ["IsAlone", "HasCabin", "IsChild", "IsElderly", "IsWomanChild", "LargeFamily", "SmallFamily"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Pclass", "Title", "CabinDeck", "TicketPrefix"]
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


def _extract_ticket_prefix(ticket: str) -> str:
    """Extract ticket prefix which can indicate passenger group/class."""
    if not isinstance(ticket, str):
        return "NUMERIC"
    ticket = ticket.strip()
    if ticket.isdigit():
        return "NUMERIC"
    # Extract prefix before numbers
    parts = ticket.split()
    if len(parts) > 1:
        return parts[0].upper().replace(".", "").replace("/", "")
    return "NUMERIC"


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    
    # Basic family features
    engineered["FamilySize"] = engineered["SibSp"].fillna(0) + engineered["Parch"].fillna(0) + 1
    engineered["IsAlone"] = (engineered["FamilySize"] == 1).astype(int)
    
    # Title extraction
    engineered["Title"] = engineered["Name"].apply(_extract_title)
    
    # Cabin features
    engineered["CabinDeck"] = engineered["Cabin"].apply(_extract_cabin_deck)
    engineered["HasCabin"] = engineered["Cabin"].notna().astype(int)
    
    # Ticket features
    engineered["TicketPrefix"] = engineered["Ticket"].apply(_extract_ticket_prefix)
    engineered["TicketGroupSize"] = (
        engineered.groupby("Ticket")["Ticket"].transform("count").clip(lower=1)
    )
    
    # Age features
    age_filled = engineered["Age"].fillna(engineered["Age"].median())
    engineered["IsChild"] = (age_filled < 16).astype(int)
    engineered["IsElderly"] = (age_filled > 60).astype(int)
    # Age groups: 0-12, 13-17, 18-25, 26-35, 36-50, 51-60, 60+
    age_groups = pd.cut(
        age_filled,
        bins=[0, 12, 17, 25, 35, 50, 60, 100],
        labels=[0, 1, 2, 3, 4, 5, 6],
        include_lowest=True
    )
    engineered["AgeGroup"] = age_groups.astype(float).fillna(3.0)  # Default to middle age group
    
    # Fare features
    fare_filled = engineered["Fare"].fillna(engineered["Fare"].median())
    engineered["FarePerPerson"] = fare_filled / engineered["FamilySize"]
    # Fare groups based on quantiles
    fare_quantiles = fare_filled.quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fare_groups = pd.cut(
        fare_filled,
        bins=fare_quantiles.values,
        labels=[0, 1, 2, 3, 4],
        include_lowest=True
    )
    engineered["FareGroup"] = fare_groups.astype(float).fillna(2.0)  # Default to middle fare group
    
    # Interaction features (very important for Titanic!)
    # Sex * Pclass (women in higher classes had much better survival)
    engineered["Sex_Pclass"] = (
        engineered["Sex"].map({"male": 0, "female": 1}) * engineered["Pclass"]
    )
    
    # Age * Pclass (children in higher classes)
    engineered["Age_Pclass"] = age_filled * engineered["Pclass"]
    
    # Fare * Pclass
    engineered["Fare_Pclass"] = fare_filled * engineered["Pclass"]
    
    # FamilySize * Pclass
    engineered["FamilySize_Pclass"] = engineered["FamilySize"] * engineered["Pclass"]
    
    # Age * Sex (women and children first)
    engineered["Age_Sex"] = age_filled * engineered["Sex"].map({"male": 0, "female": 1})
    
    # Fare * Sex
    engineered["Fare_Sex"] = fare_filled * engineered["Sex"].map({"male": 0, "female": 1})
    
    # Woman or Child (very high survival rate)
    engineered["IsWomanChild"] = (
        (engineered["Sex"] == "female") | (age_filled < 16)
    ).astype(int)
    
    # More sophisticated features
    # Family survival pattern (using family characteristics)
    engineered["LargeFamily"] = (engineered["FamilySize"] > 4).astype(int)
    engineered["SmallFamily"] = ((engineered["FamilySize"] >= 2) & (engineered["FamilySize"] <= 4)).astype(int)
    
    # Age * FamilySize interaction
    engineered["Age_FamilySize"] = age_filled * engineered["FamilySize"]
    
    # Fare per person squared (non-linear relationship)
    engineered["FarePerPersonSq"] = engineered["FarePerPerson"] ** 2
    
    # Cabin deck numeric encoding (A=1, B=2, etc.)
    deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "Unknown": 0}
    engineered["CabinDeckNumeric"] = engineered["CabinDeck"].map(deck_map).fillna(0)
    
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
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
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


def load_full_training_data(
    data_dir: Path,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """Load and preprocess full training dataset without splitting."""
    train_path = data_dir / "train.csv"
    _validate_data_path(train_path)
    raw_df = pd.read_csv(train_path)
    engineered_df = _engineer_features(raw_df)
    y = engineered_df[TARGET_COLUMN].to_numpy()
    feature_df = engineered_df[NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = _build_preprocessor()
    features = preprocessor.fit_transform(feature_df)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return features, y, preprocessor, feature_names


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

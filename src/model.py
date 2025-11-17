from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers


def build_model(
    input_dim: int,
    hidden_units: Iterable[int] = (128, 64, 32),
    dropout_rate: float = 0.3,
) -> keras.Model:
    model = keras.Sequential(name="titanic_dnn")
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def train_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
) -> keras.callbacks.History:
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.keras"
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=2,
    )
    model.save(checkpoint_path)
    return history


def export_full_model(model: keras.Model, export_path: Path) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(export_path)

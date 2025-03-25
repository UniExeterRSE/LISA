import json
import os
from pathlib import Path

import joblib
import keras_tuner as kt
import polars as pl
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR
from lisa.features import sequential_stratified_split, standard_scaler
from lisa.modeling.multipredictor import _log_parameters


def create_classifier(neurons1, neurons2, dropout):
    model = keras.models.Sequential(
        [
            keras.layers.Dense(neurons1, activation="relu", input_shape=(60,)),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(neurons2, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(3, activation="softmax"),  # 3 classes for classification
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_classifier(hp):
    neurons1 = hp.Int("units1", min_value=32, max_value=512, step=32)
    neurons2 = hp.Int("units2", min_value=32, max_value=512, step=32)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)

    model = create_classifier(neurons1=neurons1, neurons2=neurons2, dropout=dropout)
    return model


def main(input_path: Path = PROCESSED_DATA_DIR / "reduced_main_data.parquet"):
    """
    Train a neural network classifier for activity recognition.
    The output is saved to MODELS_DIR/neural_net.
    """
    df = pl.scan_parquet(input_path)

    X_train, X_val, y_train, y_val = sequential_stratified_split(df, 0.8, 800, ["ACTIVITY"])

    label_encoder = LabelEncoder()

    X_train, X_val, scaler = standard_scaler(X_train, X_val)

    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()

    y_train = y_train.collect().to_numpy()
    y_val = y_val.collect().to_numpy()

    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)

    tuner = kt.BayesianOptimization(
        build_classifier,
        objective="val_loss",
        max_trials=10,
        overwrite=True,
    )

    tuner.search(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val),
        batch_size=32,
    )

    best_model = tuner.get_best_models()[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    output_dir = MODELS_DIR / "neural_net"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_model.save(output_dir / "activity.keras")

    with open(output_dir / "scaler.pkl", "wb") as f:
        joblib.dump(scaler, f)

    output = _log_parameters(df, best_hps.values, 800, 0.8)

    output["score"]["activity"] = best_model.evaluate(X_val, y_val)[1]

    output_json_path = output_dir / "output.json"
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Output saved to: {output_json_path}")


if __name__ == "__main__":
    main()

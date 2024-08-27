from pathlib import Path

import joblib
import polars as pl
import typer
from loguru import logger
from sklearn import metrics

from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "labelled_test_data.csv",
    model_path: Path = MODELS_DIR / "logistic_regression.pkl",
    scaler_path: Path = MODELS_DIR / "logistic_regression_scaler.pkl",
):
    """
    Load model and scaler from pkl files and apply them to a new dataset.
    """
    # Load the model and scaler
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logger.info(f"Loading scaler from {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load the dataset
    logger.info(f"Loading features from {features_path}")
    features = pl.read_csv(features_path)

    X = features.select(pl.exclude(["ACTIVITY", "TRIAL", "TIME"]))
    y = features.select("ACTIVITY")

    # Apply the scaler to the dataset
    logger.info("Scaling the features")
    X = scaler.transform(X)

    # Perform predictions
    logger.info("Performing predictions")

    cm = metrics.confusion_matrix(y, model.predict(X), normalize="true")

    logger.info("Score: " + str(model.score(X, y)))
    logger.info("Confusion Matrix:\n" + str(cm))

    logger.success("Inference complete.")


if __name__ == "__main__":
    app()

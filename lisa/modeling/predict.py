from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import polars as pl
import typer
from loguru import logger
from sklearn import metrics

from lisa import evaluate
from lisa.config import MODELS_DIR

app = typer.Typer()


@app.command()
def apply_model(
    features_path: Path,
    feature: Literal["ACTIVITY", "SPEED", "INCLINE"],
    model_path: Path,
    scaler_path: Path | None = None,
) -> None:
    """
    Load a pre-trained model and scaler from pkl files and apply them to a new dataset.
    Evaluation plots are saved in MODELS_DIR/validation, and scores are saved in MODELS_DIR/validation/results.csv.

    Args:
        features_path (Path):The unseen processed dataset.
        feature (Literal["ACTIVITY", "SPEED", "INCLINE"]): The feature to predict.
        model_path (Path): Path to the pre-trained model.
        scaler_path (Path | None): Path to the pre-trained scaler; required for linear/logistic regression.
            Defaults to None.

    Returns:
        None
    """
    # Load the model and scaler, if needed
    logger.info(f"Loading model from {model_path}")
    model, column_names = joblib.load(model_path)
    if scaler_path:
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)

    # Load the dataset
    logger.info(f"Loading features from {features_path}")
    features = pl.read_parquet(features_path)

    X = features.select(pl.exclude(["ACTIVITY", "TRIAL", "TIME", "INCLINE", "SPEED"]))
    y = features.select(feature)

    X = X.select(column_names)

    if feature in ["SPEED", "INCLINE"]:
        # Filter out the rows with null values (non-locomotion)
        train_non_null_mask = y.to_series(0).is_not_null()
        X = X.filter(train_non_null_mask)
        y = y.filter(train_non_null_mask)

    if scaler_path:
        # Apply the scaler to the dataset
        logger.info("Scaling the features")
        X = scaler.transform(X)

    logger.info("Performing predictions")

    # Save the results
    RESULTS_DIR = MODELS_DIR / "validation"
    results = {
        "val_data": [features_path.stem],
        "run_id": [model_path.parent.name],
        "feature": [feature],
        "score": [model.score(X, y)],
    }

    logger.info("Score: " + str(model.score(X, y)))

    if feature == "ACTIVITY":
        labels = y["ACTIVITY"].unique(maintain_order=True)
        cm_plot_path = RESULTS_DIR / f"{model_path.parent.name}_cm.png"
        cm = evaluate.confusion_matrix(model, labels, X, y, cm_plot_path)
        logger.info("Confusion Matrix:\n" + str(cm))

        results["plot_path"] = str(cm_plot_path.stem)
        results["rmse"] = None
    else:
        results["plot_path"] = None
        results["rmse"] = np.sqrt(metrics.mean_squared_error(y, model.predict(X)))

    results_csv_path = RESULTS_DIR / "results.csv"
    if results_csv_path.exists():
        results_store = pl.read_csv(results_csv_path)
        results_store = results_store.vstack(pl.DataFrame(results))
    else:
        logger.info(f"{results_csv_path} does not exist. Creating a new file.")
        results_store = pl.DataFrame(results)

    results_store.write_csv(results_csv_path)
    logger.success("Inference complete.")


if __name__ == "__main__":
    apply_model()

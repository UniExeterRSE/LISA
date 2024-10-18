import pickle
import time
from collections.abc import Iterable
from pathlib import Path
from pickle import dump

import mlflow
import polars as pl
from loguru import logger
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lisa import evaluate
from lisa.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, MLFLOW_URI, MODELS_DIR
from lisa.features import sequential_stratified_split, sliding_window, standard_scaler


def logistic_regression(X_train: ndarray, y_train: ndarray) -> OneVsRestClassifier:
    """
    Fits a logistic regression model to the input data.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels.

    Returns:
        OneVsRestClassifier: The trained logistic regression model.
    """

    logisticRegr = OneVsRestClassifier(LogisticRegression(n_jobs=-1, random_state=42))

    logisticRegr.fit(X_train, y_train)

    return logisticRegr


def main(
    data_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv",
    windows: Iterable[int] = [300],
    splits: Iterable[float] = [0.8],
    log: bool = False,
    save: bool = False,
    model_path: Path = MODELS_DIR / "labelled_sample/logistic_regression.pkl",
):
    """
    Train a logistic regression classifier model on the data.
    Logs the score and confusion matrix, and optionally saves model to a pickle file.

    Args:
        features_path (Path): Path to the data.
        windows (Iterable[int]): List of window sizes for sliding window. Default [300].
        splits (Iterable[float]): List of train-test splits. Must be in (0, 1). Default [0.8].
        log (bool): Flag to log the model for MLflow. Default False.
        save (bool): Flag to save the model. Default False.
        model_path (Path): Path to save the trained model.
    """
    start_time = time.time()
    input_df = pl.read_csv(data_path)

    mlflow.set_tracking_uri(uri=MLFLOW_URI)

    # Create a new MLflow Experiment
    mlflow.set_experiment("LR Test")
    # Start an MLflow run
    with mlflow.start_run():
        for window in windows:
            for split in splits:
                with mlflow.start_run(nested=True, run_name=f"W_{window}:S_{split}"):
                    df = sliding_window(input_df, period=window, log=True)

                    X_train, X_test, y_train, y_test = sequential_stratified_split(df, train_size=split, gap=window)

                    scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)

                    model = logistic_regression(scaled_X_train, y_train)

                    score = model.score(scaled_X_test, y_test)

                    # save scaler and model to pickle file
                    if save:
                        with open(model_path.with_stem(model_path.stem + "_scaler"), "wb") as f:
                            dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
                        with open(model_path, "wb") as f:
                            dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.success(f"Model saved to: {model_path}")

                    # Create and log confusion matrix
                    labels = df["ACTIVITY"].unique(maintain_order=True)
                    cm_plot_path = ARTIFACTS_DIR / "confusion_matrix.png"
                    cm = evaluate.confusion_matrix(model, labels, scaled_X_test, y_test, cm_plot_path)
                    logger.info("Confusion Matrix:\n" + str(cm))
                    mlflow.log_artifact(cm_plot_path)

                    # Log the hyperparameters
                    params = {}
                    params["window"] = window
                    params["split"] = split
                    mlflow.log_params(params)

                    # Log metrics
                    mlflow.log_metric("score", score)

                    # Set a tag that we can use to remind ourselves what this run was for
                    mlflow.set_tag("Training Info", "Basic LR model for labelled test data")

                    if log:
                        # Infer the model signature
                        pd_scaled_X_train = scaled_X_train.to_pandas()
                        signature = mlflow.models.infer_signature(scaled_X_train, model.predict(pd_scaled_X_train))

                        # Log the model
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="lr_model",
                            signature=signature,
                            input_example=pd_scaled_X_train,
                        )

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

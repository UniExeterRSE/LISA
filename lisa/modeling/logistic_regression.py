import pickle
import time
from collections.abc import Iterable
from pathlib import Path
from pickle import dump

import mlflow
import polars as pl
from loguru import logger
from numpy import ndarray
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lisa import evaluate
from lisa.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, MLFLOW_URI, MODELS_DIR
from lisa.features import (
    check_split_balance,
    sequential_stratified_split,
    sliding_window,
    standard_scaler,
)
from lisa.plots import regression_histogram


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


def linear_regressor(X_train: ndarray, X_test: ndarray, y_train: ndarray, y_test: ndarray) -> tuple[ndarray, float]:
    """
    Fits a linear regression model to the input data.
    Filters out the rows with null values (non-locomotion activities) before fitting.

    Args:
        X_train (ndarray): The training data.
        X_test (ndarray): The test data.
        y_train (ndarray): The training labels.
        y_test (ndarray): The test labels.

    Returns:
        tuple[ndarray, float]: The predicted values and the model score.
    """
    model = LinearRegression(n_jobs=-1)

    # Filter out the rows with null values (non-locomotion)
    train_non_null_mask = y_train.to_series(0).is_not_null()
    X_train_filtered = X_train.filter(train_non_null_mask)
    y_train_filtered = y_train.filter(train_non_null_mask)

    model.fit(X_train_filtered, y_train_filtered)

    test_non_null_mask = y_test.to_series(0).is_not_null()
    X_test_filtered = X_test.filter(test_non_null_mask)
    y_test_filtered = y_test.filter(test_non_null_mask)
    y_pred = model.predict(X_test_filtered)

    y_score = model.score(X_test_filtered, y_test_filtered)

    return y_pred, y_score


# TODO test this works
def _regressor_script(
    feature_name: str,
    df: pl.DataFrame,
    X_train: ndarray,
    X_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
) -> tuple[float, Path]:
    """
    Script set-up and tear-down for fitting linear regressor.
    Logs any imbalance in train-test split, fits the model, and saves the histogram plot.

    Args:
        feature_name (str): The name of the feature to predict, i.e 'Speed'.
        df (pl.DataFrame): The full DataFrame.
        X_train (ndarray): The training data.
        X_test (ndarray): The test data.
        y_train (ndarray): The training labels.
        y_test (ndarray): The test labels.

    Returns:
        tuple[float, Path]: The model score and the path to the histogram plot.
    """
    if not check_split_balance(y_train, y_test).is_empty():
        logger.info(f"{feature_name} unbalance: {check_split_balance(y_train, y_test)}")

    y_pred, y_score = linear_regressor(X_train, X_test, y_train, y_test)

    y_plot_path = ARTIFACTS_DIR / f"{feature_name}_hist.png"
    hist = regression_histogram(df, y_pred, feature_name)

    hist.savefig(y_plot_path)

    return y_score, y_plot_path


def multipredictor(
    data_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv",
    window: int = 300,
    split: float = 0.8,
):
    """
    Runs a multimodel predictor on the input data.
    Classifies activity, and predicts speed and incline.
    Three separate models & scores are trained and logged to MLflow.

    Args:
        data_path (Path): Path to the data.
        window (int): Size of the sliding window. Default 300.
        split (float): Train-test split. Default 0.8.
    """
    start_time = time.time()
    input_df = pl.read_csv(data_path)

    mlflow.set_tracking_uri(uri=MLFLOW_URI)

    # Create a new MLflow Experiment
    mlflow.set_experiment("LR multipredictor test")
    # Start an MLflow run
    with mlflow.start_run():
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "LR Multipredictor development")

        # Prepare data
        df = sliding_window(input_df, period=window, log=True)
        X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = sequential_stratified_split(
            df, split, window, ["ACTIVITY", "SPEED", "INCLINE"]
        )
        scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)

        # Log the hyperparameters
        params = {}
        params["window"] = window
        params["split"] = split
        mlflow.log_params(params)
        # TODO move each prediction into separate functions
        # Predict activity
        with mlflow.start_run(nested=True, run_name="activity classifier"):
            if not check_split_balance(y1_train, y1_test).is_empty():
                logger.info(f"Activity unbalance: {check_split_balance(y1_train, y1_test)}")

            activity_model = logistic_regression(scaled_X_train, y1_train)
            y1_score = activity_model.score(scaled_X_test, y1_test)
            mlflow.log_metric("score", y1_score)

            # Create and log confusion matrix
            labels = df["ACTIVITY"].unique(maintain_order=True)
            cm_plot_path = ARTIFACTS_DIR / "LR_confusion_matrix.png"
            cm = evaluate.confusion_matrix(activity_model, labels, scaled_X_test, y1_test, cm_plot_path)
            logger.info("Confusion Matrix:\n" + str(cm))
            mlflow.log_artifact(cm_plot_path)

        # Predict speed
        with mlflow.start_run(nested=True, run_name="speed regressor"):
            y2_score, y2_plot_path = _regressor_script("Speed", df, scaled_X_train, scaled_X_test, y2_train, y2_test)
            mlflow.log_metric("score", y2_score)
            mlflow.log_artifact(y2_plot_path)

        # Predict incline
        with mlflow.start_run(nested=True, run_name="incline regressor"):
            y3_score, y3_plot_path = _regressor_script("Incline", df, scaled_X_train, scaled_X_test, y3_train, y3_test)
            mlflow.log_metric("score", y3_score)
            mlflow.log_artifact(y3_plot_path)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


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

import json
import re
import time
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl
from loguru import logger
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lisa import evaluate
from lisa.config import (
    ARTIFACTS_DIR,
    FOOT_SENSOR_PATTERN,
    IMU_PATTERN,
    MLFLOW_URI,
    PROCESSED_DATA_DIR,
)
from lisa.features import (
    check_split_balance,
    sequential_stratified_split,
    standard_scaler,
)
from lisa.plots import regression_histogram

# Define type aliases
ClassifierModel = OneVsRestClassifier | RandomForestClassifier | lgb.LGBMClassifier
TreeBasedRegressorModel = RandomForestRegressor | lgb.LGBMRegressor
RegressorModel = LinearRegression | TreeBasedRegressorModel


def classifier(model_name: str, X_train: ndarray, y_train: ndarray, params: dict[str, any]) -> ClassifierModel:
    """
    Fits a classifier model to the input data.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels.
        params (dict[str, any]): The hyperparameters for the model.

    Returns:
        ClassifierModel: The trained classifier model.
    """
    params = params.copy()
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    models = {
        "LR": lambda **params: OneVsRestClassifier(LogisticRegression(**params)),
        "RF": lambda **params: RandomForestClassifier(**params),
        "LGBM": lambda **params: lgb.LGBMClassifier(**params),
    }

    return models[model_name](**params).fit(X_train, y_train)


def regressor(
    model_name: str,
    X_train: ndarray,
    X_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    params: dict[str, any],
) -> tuple[ndarray, float, RegressorModel]:
    """
    Fits a regressor model to the input data.
    Filters out the rows with null values (non-locomotion activities) before fitting.

    Args:
        X_train (ndarray): The training data.
        X_test (ndarray): The test data.
        y_train (ndarray): The training labels.
        y_test (ndarray): The test labels.
        params (dict[str, any]): The hyperparameters for the model.

    Returns:
        tuple[ndarray, float, RegressorModel]: The predicted value, model score and model.
    """

    params = params.copy()
    if model_name != "LR":
        params.setdefault("random_state", 42)

    if model_name == "RF":
        params.setdefault("n_estimators", 10)
        params.setdefault("max_depth", 10)

    params.setdefault("n_jobs", -1)

    models = {
        "LR": lambda **params: LinearRegression(**params),
        "RF": lambda **params: RandomForestRegressor(**params),
        "LGBM": lambda **params: lgb.LGBMRegressor(**params),
    }
    model = models[model_name](**params)

    # Filter out the rows with null values (non-locomotion)
    train_non_null_mask = y_train.to_series(0).is_not_null()
    X_train_filtered = X_train.filter(train_non_null_mask)
    y_train_filtered = y_train.filter(train_non_null_mask)

    model.fit(X_train_filtered, y_train_filtered.to_numpy().ravel())

    test_non_null_mask = y_test.to_series(0).is_not_null()
    X_test_filtered = X_test.filter(test_non_null_mask)
    y_test_filtered = y_test.filter(test_non_null_mask)
    y_pred = model.predict(X_test_filtered)

    y_score = model.score(X_test_filtered, y_test_filtered)

    return y_pred, y_score, model


def _regressor_script(
    model_name: str,
    feature_name: str,
    df: pl.DataFrame,
    X_train: ndarray,
    X_test: ndarray,
    y_train: ndarray,
    y_test: ndarray,
    hyperparams: dict[str, any],
) -> tuple[float, Path, Path]:
    """
    Script set-up and tear-down for fitting the regressor model.
    Logs any imbalance in train-test split, fits the model, and saves the histogram plot
    and feature importances.

    Args:
        feature_name (str): The name of the feature to predict, i.e 'Speed'.
        df (pl.DataFrame): The full DataFrame.
        X_train (ndarray): The training data.
        X_test (ndarray): The test data.
        y_train (ndarray): The training labels.
        y_test (ndarray): The test labels.
        hyperparams (dict[str, any]): The hyperparameters for the model.

    Returns:
        tuple[float, Path, Path | None]: The model score, path to the histogram plot,
        and path to the feature importances if it exists.
    """
    if not check_split_balance(y_train.lazy(), y_test.lazy()).is_empty():
        logger.info(f"{feature_name} unbalance: {check_split_balance(y_train.lazy(), y_test.lazy())}")

    y_pred, y_score, model = regressor(model_name, X_train, X_test, y_train, y_test, hyperparams)

    y_plot_path = ARTIFACTS_DIR / f"{model_name}_{feature_name}_hist.png"
    hist = regression_histogram(df, y_pred, feature_name.upper())

    hist.savefig(y_plot_path)

    feature_importances_path = None
    if model_name == "LGBM" or model_name == "RF":
        sorted_feature_importance_dict = _feature_importances(model, X_train)

        feature_importances_path = ARTIFACTS_DIR / f"feature_importances_{model}_{feature_name}.json"
        with open(feature_importances_path, "w") as f:
            json.dump(sorted_feature_importance_dict, f, indent=4)

    return y_score, y_plot_path, feature_importances_path


def _feature_importances(model: TreeBasedRegressorModel, X_train: pl.DataFrame) -> dict[str, float]:
    """
    Extracts and logs the feature importances from the model.

    Args:
        model (TreeBasedRegressorModel): The trained model.
        X_train (pl.DataFrame): The training data.

    Returns:
        dict[str, float]: The sorted feature importances.
    """
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    feature_names = X_train.columns
    feature_importance_dict = {
        feature_names[indices[i]]: float(feature_importances[indices[i]]) for i in range(len(feature_importances))
    }
    return dict(
        sorted(
            feature_importance_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def main(
    data_path: Path = PROCESSED_DATA_DIR / "P5.parquet",
    model: Literal["LR", "RF", "LGBM"] = "RF",
    window: int = 300,
    split: float = 0.8,
):
    """
    Runs a multimodel predictor on the input data.
    Classifies activity, and predicts speed and incline.
    Three separate models & scores are trained and logged to MLflow.

    Args:
        data_path (Path): Path to the data.
        model (Literal["LR", "RF", "LGBM"]): Short name of the model 'family' to use.
            Currently supports 'LR' (logistic/linear regression), 'RF' (random forest), 'LGBM' (LightGBM).
        window (int): Size of the sliding window. Default 300.
        split (float): Train-test split. Default 0.8.
    """
    start_time = time.time()

    input_df = pl.scan_parquet(data_path)

    # Prepare data
    df = input_df
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = sequential_stratified_split(
        df, split, window, ["ACTIVITY", "SPEED", "INCLINE"]
    )
    logger.info("scaling data...")
    scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)
    # scaled_X_train, scaled_X_test = X_train, X_test
    logger.info("data scaled")

    mlflow.set_tracking_uri(uri=MLFLOW_URI)
    mlflow.set_experiment(f"{model} bugfixing")  # Create a new MLflow Experiment
    with mlflow.start_run(run_name=data_path.stem, log_system_metrics=True):
        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", f"{model} Multipredictor development")

        # Extract the unique components from the column names to log
        statistic, measure, location, dimension = set(), set(), set(), set()

        imu_pattern = re.compile(IMU_PATTERN)
        foot_sensor_pattern = re.compile(FOOT_SENSOR_PATTERN)

        for key in df.collect_schema().names():
            imu_match = imu_pattern.match(key)
            foot_sensor_match = foot_sensor_pattern.match(key)
            if imu_match:
                stat, meas, loc, dim = imu_match.groups()
                statistic.add(stat)
                measure.add(meas)
                location.add(loc)
                dimension.add(dim)
            elif foot_sensor_match:
                stat, loc = foot_sensor_match.groups()
                statistic.add(stat)
                location.add(loc)

        # Log the hyperparameters
        params = {
            "window": window,
            "split": split,
            "statistic": statistic,
            "measure": measure,
            "location": location,
            "dimension": dimension,
        }
        mlflow.log_params(params)

        hyperparams = {}

        # Predict activity
        with mlflow.start_run(nested=True, run_name="activity classifier"):
            if not check_split_balance(y1_train, y1_test).is_empty():
                logger.info(f"Activity unbalance: {check_split_balance(y1_train, y1_test)}")

            # TODO collect all data, for now
            y1_train = y1_train.collect()
            y1_test = y1_test.collect()
            y2_train = y2_train.collect()
            y2_test = y2_test.collect()
            y3_train = y3_train.collect()
            y3_test = y3_test.collect()
            df = df.collect()

            activity_model = classifier(
                model,
                scaled_X_train,
                y1_train.to_numpy().ravel(),
                hyperparams,
            )

            y1_score = activity_model.score(scaled_X_test, y1_test)
            mlflow.log_metric("score", y1_score)

            # Create and log confusion matrix
            labels = df["ACTIVITY"].unique(maintain_order=True)
            cm_plot_path = ARTIFACTS_DIR / f"{model}_confusion_matrix.png"
            cm = evaluate.confusion_matrix(activity_model, labels, scaled_X_test, y1_test, cm_plot_path)
            logger.info("Confusion Matrix:\n" + str(cm))
            mlflow.log_artifact(cm_plot_path)

        # Predict speed
        with mlflow.start_run(nested=True, run_name="speed regressor"):
            y2_score, y2_plot_path, feature_importances_path = _regressor_script(
                model,
                "Speed",
                df,
                scaled_X_train,
                scaled_X_test,
                y2_train,
                y2_test,
                hyperparams,
            )

            mlflow.log_metric("score", y2_score)
            mlflow.log_artifact(y2_plot_path)
            if feature_importances_path:
                mlflow.log_artifact(feature_importances_path)

        # Predict incline
        with mlflow.start_run(nested=True, run_name="incline regressor"):
            y3_score, y3_plot_path, feature_importances_path = _regressor_script(
                model,
                "Incline",
                df,
                scaled_X_train,
                scaled_X_test,
                y3_train,
                y3_test,
                hyperparams,
            )

            mlflow.log_metric("score", y3_score)
            mlflow.log_artifact(y3_plot_path)
            if feature_importances_path:
                mlflow.log_artifact(feature_importances_path)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

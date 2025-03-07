import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Literal

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from loguru import logger
from numpy import ndarray
from sklearn import metrics, set_config
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from lisa import evaluate
from lisa.config import FOOT_SENSOR_PATTERN, IMU_PATTERN, MODELS_DIR, PROJ_ROOT
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


def classifier(model_name: str, X_train: pl.DataFrame, y_train: pl.Series, params: dict[str, any]) -> ClassifierModel:
    """
    Fits a classifier model to the input data.

    Args:
        X_train (pl.DataFrame): The training data.
        y_train (pl.Series): The training labels.
        params (dict[str, any]): The hyperparameters for the model.

    Returns:
        ClassifierModel: The trained classifier model.
    """
    # allow sample_weight in the fit method for LR
    set_config(enable_metadata_routing=True)

    params = params.copy()
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    # Calculate class weights in training data
    value_counts = y_train.value_counts(normalize=True)
    class_weights = {row[0]: 1 / row[1] for row in value_counts.iter_rows()}
    sample_weight = np.array([class_weights[label] for label in y_train])

    models = {
        "LR": lambda **params: OneVsRestClassifier(LogisticRegression(**params).set_fit_request(sample_weight=True)),
        "RF": lambda **params: RandomForestClassifier(**params).set_fit_request(sample_weight=True),
        "LGBM": lambda **params: lgb.LGBMClassifier(**params).set_fit_request(sample_weight=True),
    }

    return models[model_name](**params).fit(X_train, y_train, sample_weight=sample_weight)


def regressor(
    model_name: str,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    params: dict[str, any],
) -> tuple[pl.DataFrame, ndarray, RegressorModel]:
    """
    Fits a regressor model to the input data.
    Filters out the rows with null values (non-locomotion activities) before fitting.

    Args:
        X_train (pl.DataFrame): The training data.
        X_test (pl.DataFrame): The test data.
        y_train (pl.DataFrame): The training labels.
        y_test (pl.DataFrame): The test labels.
        params (dict[str, any]): The hyperparameters for the model.

    Returns:
        tuple[pl.DataFrame, ndarray, RegressorModel]: The true values, predicted values, and model.
    """

    params = params.copy()
    if model_name != "LR":
        params.setdefault("random_state", 42)
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

    return y_test_filtered, y_pred, model


def _regressor_script(
    model_name: str,
    feature_name: str,
    X_train: pl.DataFrame,
    X_test: pl.DataFrame,
    y_train: pl.DataFrame,
    y_test: pl.DataFrame,
    hyperparams: dict[str, any],
    output_dir: Path,
) -> tuple[float, float, RegressorModel]:
    """
    Script set-up and tear-down for fitting the regressor model.
    Logs any imbalance in train-test split, fits the model, and saves the histogram plot
    and feature importances.

    Args:
        feature_name (str): The name of the feature to predict, i.e 'Speed'.
        X_train (pl.DataFrame): The training data.
        X_test (pl.DataFrame): The test data.
        y_train (pl.DataFrame): The training labels.
        y_test (pl.DataFrame): The test labels.
        hyperparams (dict[str, any]): The hyperparameters for the model.

    Returns:
        float: The r2 score.
        float: The rmse score.
        RegressorModel: The trained regressor model.
    """
    if not check_split_balance(y_train.lazy(), y_test.lazy()).is_empty():
        logger.info(f"{feature_name} unbalance: {check_split_balance(y_train.lazy(), y_test.lazy())}")

    y_test_filtered, y_pred, model = regressor(model_name, X_train, X_test, y_train, y_test, hyperparams)

    rmse = np.sqrt(metrics.mean_squared_error(y_test_filtered, y_pred))
    r2 = metrics.r2_score(y_test_filtered, y_pred)

    y_plot_path = output_dir / f"{feature_name}_hist.png"
    hist = regression_histogram(y_test_filtered, y_pred, feature_name.upper())

    hist.savefig(y_plot_path)

    if model_name == "LGBM" or model_name == "RF":
        sorted_feature_importance_dict = _feature_importances(model, X_train)

        feature_importances_path = output_dir / f"feature_importances_{feature_name}.json"
        with open(feature_importances_path, "w") as f:
            json.dump(sorted_feature_importance_dict, f, indent=4)

    return r2, rmse, model


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


def _log_parameters(df: pl.DataFrame, hyperparams: dict[str, any], window: int, split: float) -> dict[str, any]:
    """
    Logs the parameters used in the models.

    Args:
        df (pl.DataFrame): The input data.
        hyperparams (dict[str, any]): The tuning hyperparameters for the models.
        window (int): The size of the sliding window.
        split (float): The train-test split.

        Returns:
        dict[str, any]: The output dictionary.
    """

    # Initialise the output dictionary, including scores
    output = {
        "score": {
            "activity": None,
            "activity_weighted": None,
            "speed_r2": None,
            "speed_rmse": None,
            "incline_r2": None,
            "incline_rmse": None,
        },
        "params": {},
    }

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

    # Log the parameters
    output["params"] = {
        "window": window,
        "split": split,
        "statistic": list(statistic),
        "measure": list(measure),
        "location": list(location),
        "dimension": list(dimension),
        "hyperparams": hyperparams,
    }

    return output


def _save_output(
    output: dict,
    output_dir: Path,
    scaler: StandardScaler | None,
    activity_model: ClassifierModel,
    speed_model: RegressorModel,
    incline_model: RegressorModel,
    scaled_X_train: pl.DataFrame,
    save: bool,
) -> None:
    """
    Save the output to a JSON file and optionally save the scaler and models to pickle files.

    Args:
        output (dict): The output dictionary.
        output_dir (Path): Directory to save the files.
        scaler (standardScaler | None): The scaler object, if one exists.
        activity_model (ClassifierModel): The activity model.
        speed_model (RegressorModel): The speed model.
        incline_model (RegressorModel): The incline model.
        scaled_X_train (pl.DataFrame): The scaled training data.
        save (bool): Whether to save the scaler and models to pickle files.
    """
    # Save output to a JSON file
    output_json_path = output_dir / "output.json"
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)
    logger.info(f"Output saved to: {output_json_path}")

    # Save scaler and models to pickle files
    if save:
        if scaler is not None:
            with open(output_dir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info("Scaler saved to pickle file")
        with open(output_dir / "activity.pkl", "wb") as f:
            joblib.dump((activity_model, scaled_X_train.columns), f)
        with open(output_dir / "speed.pkl", "wb") as f:
            joblib.dump((speed_model, scaled_X_train.columns), f)
        with open(output_dir / "incline.pkl", "wb") as f:
            joblib.dump((incline_model, scaled_X_train.columns), f)
        logger.info("Models saved to pickle files")


def multipredictor(
    data_path: Path,
    run_name: str,
    model: Literal["LR", "RF", "LGBM"],
    window: int = 800,
    split: float = 0.8,
    save: bool = False,
):
    """
    Runs a multimodel predictor on the input data.
    Classifies activity, and predicts speed and incline.
    Three separate models are trained, validated and logged.

    Args:
        data_path (Path): Path to the data parquet file.
        run_name (str): Name of the run.
        model (Literal["LR", "RF", "LGBM"]): Short name of the model 'family' to use.
            Currently supports 'LR' (logistic/linear regression), 'RF' (random forest), 'LGBM' (LightGBM).
        window (int): Size of the sliding window. Default 800.
        split (float): Train-test split. Default 0.8.
        save (bool): Whether to save the scaler and mdodels to pkl files. Default False.
    """
    start_time = time.time()

    # Lazy load the data
    df = pl.scan_parquet(data_path)

    # Split the data
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = sequential_stratified_split(
        df, split, window, ["ACTIVITY", "SPEED", "INCLINE"]
    )

    # Scale the data, if necessary
    if model == "LR":
        logger.info("scaling data...")
        scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)
        logger.info("data scaled")
    else:
        scaled_X_train, scaled_X_test = X_train.collect(), X_test.collect()
        scaler = None

    # Get the hyperparameters for the model
    hyperparams_path = Path(PROJ_ROOT / "lisa" / "modeling" / "hyperparameters.json")
    with hyperparams_path.open("r") as f:
        hyperparameters = json.load(f)
    hyperparams = hyperparameters[model]

    # Create output directory
    output_dir = MODELS_DIR / run_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = _log_parameters(df, hyperparams, window, split)

    # === Predict activity ===
    if not check_split_balance(y1_train, y1_test).is_empty():
        logger.info(f"Activity unbalance: {check_split_balance(y1_train, y1_test)}")

    y1_train = y1_train.collect()
    y1_test = y1_test.collect()

    activity_model = classifier(
        model,
        scaled_X_train,
        y1_train.to_series(),
        hyperparams,
    )

    y1_score = activity_model.score(scaled_X_test, y1_test)
    output["score"]["activity"] = y1_score

    # Calculate and log the weighted f1_score
    y1_pred = activity_model.predict(scaled_X_test)
    f1_av = metrics.f1_score(y1_test, y1_pred, average="weighted")
    output["score"]["activity_weighted"] = f1_av

    # Create and log confusion matrix
    labels = df.select("ACTIVITY").collect().to_series().unique(maintain_order=True)
    cm_plot_path = output_dir / "confusion_matrix.png"
    cm = evaluate.confusion_matrix(activity_model, labels, scaled_X_test, y1_test, cm_plot_path)
    logger.info("Confusion Matrix:\n" + str(cm))

    # === Predict speed ===
    y2_train = y2_train.collect()
    y2_test = y2_test.collect()

    output["score"]["speed_r2"], output["score"]["speed_rmse"], speed_model = _regressor_script(
        model,
        "Speed",
        scaled_X_train,
        scaled_X_test,
        y2_train,
        y2_test,
        hyperparams,
        output_dir,
    )

    # === Predict incline ===
    y3_train = y3_train.collect()
    y3_test = y3_test.collect()

    output["score"]["incline_r2"], output["score"]["incline_rmse"], incline_model = _regressor_script(
        model,
        "Incline",
        scaled_X_train,
        scaled_X_test,
        y3_train,
        y3_test,
        hyperparams,
        output_dir,
    )

    # Save final outputs
    _save_output(
        output,
        output_dir,
        scaler,
        activity_model,
        speed_model,
        incline_model,
        scaled_X_train,
        save,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")

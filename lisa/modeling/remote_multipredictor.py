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
from scipy.stats import randint
from sklearn import metrics, set_config
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier

from lisa import evaluate
from lisa.config import FOOT_SENSOR_PATTERN, IMU_PATTERN, MODELS_DIR, PROCESSED_DATA_DIR
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
    # allow sample_weight in the fit method for LR
    set_config(enable_metadata_routing=True)

    params = params.copy()
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    class_weights = {"run": 1 / 0.4, "jump": 1 / 0.024, "walk": 1 / 0.576}
    sample_weight = np.array([class_weights[label] for label in y_train])

    # RF tuning
    param_grid_classifier = {
        "n_estimators": randint(10, 50),  # Number of trees
        "max_depth": [None, 10, 20, 30],  # Tree depth
        "min_samples_split": randint(2, 20),  # Minimum samples to split
        "min_samples_leaf": randint(1, 10),  # Minimum samples per leaf
        "max_features": ["sqrt", "log2"],  # Features considered at each split
        "bootstrap": [True, False],  # Use bootstrap samples
    }

    scorer = metrics.make_scorer(metrics.f1_score, average="macro", pos_label=None)

    models = {
        "LR": lambda **params: OneVsRestClassifier(LogisticRegression(**params).set_fit_request(sample_weight=True)),
        "RF": lambda **params: RandomForestClassifier(**params).set_fit_request(sample_weight=True),
        "LGBM": lambda **params: lgb.LGBMClassifier(**params).set_fit_request(sample_weight=True),
    }

    grid_search = RandomizedSearchCV(
        estimator=models[model_name](**params),
        param_distributions=param_grid_classifier,
        n_iter=50,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1,  # Use all available processors
        random_state=42,
    )

    return grid_search.fit(X_train, y_train, sample_weight=sample_weight)

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

    if model_name == "RF":
        params.setdefault("n_estimators", 10)
        params.setdefault("max_depth", 10)

    params.setdefault("n_jobs", -1)

    param_grid_regressor = {
        "n_estimators": randint(10, 50),
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": randint(2, 20),  # Minimum samples to split
        "min_samples_leaf": randint(1, 10),  # Minimum samples per leaf
        "max_features": ["sqrt", "log2"],  # Features considered at each split
        "bootstrap": [True, False],  # Use bootstrap samples
    }

    models = {
        "LR": lambda **params: LinearRegression(**params),
        "RF": lambda **params: RandomForestRegressor(**params),
        "LGBM": lambda **params: lgb.LGBMRegressor(**params),
    }
    model = models[model_name](**params)

    random_search_regressor = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid_regressor,
        n_iter=50,
        scoring="r2",
        cv=5,
        n_jobs=-1,
        verbose=2,
        random_state=42,
    )

    # Filter out the rows with null values (non-locomotion)
    train_non_null_mask = y_train.to_series(0).is_not_null()
    X_train_filtered = X_train.filter(train_non_null_mask)
    y_train_filtered = y_train.filter(train_non_null_mask)

    random_search_regressor.fit(X_train_filtered, y_train_filtered.to_numpy().ravel())

    test_non_null_mask = y_test.to_series(0).is_not_null()
    X_test_filtered = X_test.filter(test_non_null_mask)
    y_test_filtered = y_test.filter(test_non_null_mask)

    y_pred = random_search_regressor.predict(X_test_filtered)

    return y_test_filtered, y_pred, random_search_regressor


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


def main(
    data_path: Path = PROCESSED_DATA_DIR / "P1.parquet",
    run_name: str = "testing",
    model: Literal["LR", "RF", "LGBM"] = "LGBM",
    window: int = 800,
    split: float = 0.8,
    save: bool = False,
):
    """
    Runs a multimodel predictor on the input data.
    Classifies activity, and predicts speed and incline.
    Three separate models & scores are trained and logged.

    Args:
        data_path (Path): Path to the data.
        run_name (str): Name of the run. Default 'multimodel'.
        model (Literal["LR", "RF", "LGBM"]): Short name of the model 'family' to use.
            Currently supports 'LR' (logistic/linear regression), 'RF' (random forest), 'LGBM' (LightGBM).
        window (int): Size of the sliding window. Default 300.
        split (float): Train-test split. Default 0.8.
        save (bool): Whether to save the scaler and mdodels to pkl files. Default False.
    """
    start_time = time.time()

    # Create output record
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
    output_dir = MODELS_DIR / run_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_df = pl.scan_parquet(data_path)

    # Prepare data
    df = input_df
    X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = sequential_stratified_split(
        df, split, window, ["ACTIVITY", "SPEED", "INCLINE"]
    )

    if model == "LR":
        logger.info("scaling data...")
        scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)
        logger.info("data scaled")
    else:
        scaled_X_train, scaled_X_test = X_train.collect(), X_test.collect()
        scaler = None

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

    hyperparams = {}
    if model == "LGBM":
        hyperparams = {
            "bagging_fraction": 0.6,
            "bagging_freq": 5,
            "extra_trees": True,
            "feature_fraction": 0.6,
            "lambda_l1": 0.0,
            "lambda_l2": 0.1,
            "max_bin": 255,
            "max_depth": 3,
            "min_data_in_leaf": 50,
            "min_gain_to_split": 0.1,
            "min_sum_hessian_in_leaf": 0.1,
            "num_leaves": 63,
            "path_smooth": 0.3,
        }

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

    # Predict activity
    if not check_split_balance(y1_train, y1_test).is_empty():
        logger.info(f"Activity unbalance: {check_split_balance(y1_train, y1_test)}")

    # Realise the data
    y1_train = y1_train.collect()
    y1_test = y1_test.collect()
    y2_train = y2_train.collect()
    y2_test = y2_test.collect()
    y3_train = y3_train.collect()
    y3_test = y3_test.collect()

    activity_model = classifier(
        model,
        scaled_X_train,
        y1_train.to_numpy().ravel(),
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

    # Predict speed
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

    # Predict incline
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

    # Save output to a JSON file
    output_json_path = output_dir / "output.json"
    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Output saved to: {output_json_path}")

    # save scaler and models to pickle files
    if save:
        if scaler is not None:
            with open(output_dir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(output_dir / "activity.pkl", "wb") as f:
            joblib.dump((activity_model, scaled_X_train.columns), f)
        with open(output_dir / "speed.pkl", "wb") as f:
            joblib.dump((speed_model, scaled_X_train.columns), f)
        with open(output_dir / "incline.pkl", "wb") as f:
            joblib.dump((incline_model, scaled_X_train.columns), f)

        logger.info("Scaler and models saved to pickle files")

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

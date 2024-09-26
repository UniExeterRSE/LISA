import gc
import json
import pickle
import time
from collections.abc import Iterable
from pathlib import Path
from pickle import dump

import mlflow
import numpy as np
import polars as pl
from loguru import logger
from mlflow.models import infer_signature
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from lisa import evaluate
from lisa.config import ARTIFACTS_DIR, INTERIM_DATA_DIR, MLFLOW_URI, MODELS_DIR
from lisa.features import sliding_window, standard_scaler, train_test_split


def random_forest_classifier(
    X_train: np.ndarray, y_train: np.ndarray, **params: dict[str, any]
) -> RandomForestClassifier:
    """
    Fits a random forest classifier model to the input data.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.

    Returns:
        RandomForestClassifier: The trained random forest classifier model.
    """
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 42)

    rf = RandomForestClassifier(**params)

    rf.fit(X_train, y_train)

    return rf


def main(
    features_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv",
    n_iter: int = 1,
    windows: Iterable[int] = [300],
    splits: Iterable[float] = [0.8],
    log: bool = False,
    save: bool = False,
    model_path: Path = MODELS_DIR / "labelled_sample/random_forest_tuned.pkl",
):
    """
    Train a random forest classifier model on the data.
    Performs hyperparameter tuning with random search cv if n_iter > 1.
    Logs the score and confusion matrix, and optionally saves model to a pickle file.

    Args:
        features_path (Path): Path to the data.
        n_iter (int): Number of iterations for hyperparameter tuning. If 1, uses default hyperparameters with no tuning.
        windows (Iterable[int]): List of window sizes for sliding window. Default [300].
        splits (Iterable[float]): List of train-test splits. Must be in (0, 1). Default [0.8].
            If n_iter > 1, no split is performed.
        log (bool): Flag to log the model for MLflow. Default False.
        save (bool): Flag to save the model. Default False.
        model_path (Path): Path to save the trained model.
    """
    start_time = time.time()
    input_df = pl.read_csv(features_path)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri=MLFLOW_URI)

    # Create a new MLflow Experiment
    mlflow.set_experiment("RF Test2")

    if n_iter == 1:
        hyperparams = {"n_estimators": 100, "max_depth": 128}
    else:
        hyperparams = {"n_estimators": randint(50, 100), "max_depth": randint(10, 100)}
        splits = [1.0]  # test set not needed for cross validation

    # Start an MLflow run
    with mlflow.start_run():
        for window in windows:
            for split in splits:
                with mlflow.start_run(nested=True, run_name=f"W_{window}:S_{split}"):
                    df = sliding_window(input_df, period=window, log=True)

                    X_train, X_test, y_train, y_test = train_test_split(df, train_size=split, gap=window)

                    if n_iter == 1:
                        X_train, X_test, scaler = standard_scaler(X_train, X_test)
                        model = random_forest_classifier(X_train, y_train.to_numpy().ravel(), **hyperparams)
                        params = hyperparams.copy()

                        score = model.score(X_test, y_test)

                        # log confusion matrix
                        labels = df["ACTIVITY"].unique(maintain_order=True)
                        cm_plot_path = ARTIFACTS_DIR / "confusion_matrix.png"
                        cm = evaluate.confusion_matrix(model, labels, X_test, y_test, cm_plot_path)
                        logger.info("Confusion Matrix:\n" + str(cm))
                        mlflow.log_artifact(cm_plot_path)

                    else:
                        rf = RandomForestClassifier()
                        # Use random search to find the best hyperparameters
                        rand_search = RandomizedSearchCV(
                            rf,
                            param_distributions=hyperparams,
                            n_iter=n_iter,
                            cv=2,
                            verbose=3,
                            n_jobs=-1,
                            pre_dispatch=1,
                            random_state=42,
                        )
                        rand_search.fit(X_train, y_train.to_numpy().ravel())

                        params = rand_search.best_params_

                        model = rand_search.best_estimator_

                        score = rand_search.best_score_

                    # save scaler and model to pickle file
                    if save:
                        with open(model_path.with_stem(model_path.stem + "_scaler"), "wb") as f:
                            dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
                        with open(model_path, "wb") as f:
                            dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.success(f"Model saved to: {model_path}")

                    logger.info("Score: " + str(score))

                    # Log the hyperparameters
                    params["split"] = split
                    params["window"] = window
                    mlflow.log_params(params)

                    # Log metrics
                    mlflow.log_metric("score", score)

                    # Extract and log feature importances
                    feature_importances = model.feature_importances_
                    indices = np.argsort(feature_importances)[::-1]
                    feature_names = X_train.columns
                    feature_importance_dict = {
                        feature_names[indices[i]]: feature_importances[indices[i]]
                        for i in range(len(feature_importances))
                    }
                    sorted_feature_importance_dict = dict(
                        sorted(
                            feature_importance_dict.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    )

                    feature_importances_path = ARTIFACTS_DIR / "feature_importances.json"
                    with open(feature_importances_path, "w") as f:
                        json.dump(sorted_feature_importance_dict, f, indent=4)
                    mlflow.log_artifact(feature_importances_path)

                    # Set a tag that we can use to remind ourselves what this run was for
                    mlflow.set_tag("Training Info", "Basic RF model for labelled test data")

                    if log:
                        # Infer the model signature
                        pd_X_train = X_train.to_pandas()
                        signature = infer_signature(pd_X_train, model.predict(pd_X_train))

                        # # Log the model
                        mlflow.sklearn.log_model(
                            sk_model=model,
                            artifact_path="rf_model",
                            signature=signature,
                            input_example=pd_X_train,
                        )

                    # Explicitly delete objects to free memory
                    del df, X_train, X_test, y_train, y_test, model
                    gc.collect()  # Run garbage collection

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

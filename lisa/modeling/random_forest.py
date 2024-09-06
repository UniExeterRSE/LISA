import json
import pickle
import time
from pathlib import Path
from pickle import dump

import mlflow
import numpy as np
import polars as pl
import typer
from loguru import logger
from mlflow.models import infer_signature
from scipy.stats import randint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from lisa import evaluate
from lisa.config import (
    ARTIFACTS_DIR,
    INTERIM_DATA_DIR,
    MLFLOW_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from lisa.features import sliding_window, standard_scaler, train_test_split

app = typer.Typer()


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


def hyperparam_tuning(input_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv"):
    original_df = pl.read_csv(input_path)

    mlflow.set_tracking_uri(uri=MLFLOW_URI)

    # Create a new MLflow Experiment
    mlflow.set_experiment("RF Test")
    window = 300
    split = 1.0
    with mlflow.start_run(nested=True, run_name=f"W_{window}:S_{split}"):
        #  Feature engineering with params
        df = sliding_window(original_df, period=window, log=True)

        X_train, _, y_train, _ = train_test_split(df, train_size=split, gap=window)

        # Tune model
        param_dist = {"n_estimators": randint(50, 100), "max_depth": randint(10, 100)}

        rf = RandomForestClassifier()
        # Use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=1,
            cv=2,
            verbose=3,
            n_jobs=-1,
            pre_dispatch=1,
            random_state=42,
        )
        rand_search.fit(X_train, y_train.to_numpy().ravel())

        model = rand_search.best_estimator_

        # Print the best hyperparameters
        logger.info("Best hyperparameters:", rand_search.best_params_)
        params = rand_search.best_params_

        # Log the hyperparameters
        params["window"] = window
        params["split"] = split
        mlflow.log_params(params)

        # Extract and log feature importances
        # TODO analyse feature importance statistics
        feature_importances = model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]
        feature_names = X_train.columns
        feature_importance_dict = {
            feature_names[indices[i]]: feature_importances[indices[i]] for i in range(len(feature_importances))
        }
        sorted_feature_importance_dict = dict(
            sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        feature_importances_path = ARTIFACTS_DIR / "feature_importances.json"
        with open(feature_importances_path, "w") as f:
            json.dump(sorted_feature_importance_dict, f, indent=4)
        mlflow.log_artifact(feature_importances_path)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Tuned RF model for labelled test data")

        # Infer the model signature
        pd_X_train = X_train.to_pandas()
        signature = infer_signature(pd_X_train, model.predict(pd_X_train))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="rf_model",
            signature=signature,
            input_example=pd_X_train,
        )


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "labelled_test_data.csv",
    model_path: Path = MODELS_DIR / "labelled_sample/random_forest_tuned.pkl",
    save: bool = typer.Option(False, help="Flag to save the model"),
):
    """
    Train a random forest classifier model on the processed data.
    Logs the score and confusion matrix, and optionally saves model to a pickle file.

    Args:
        features_path (Path): Path to the processed data.
        model_path (Path): Path to save the trained model.
        save (bool): Flag to save the model.
    """
    start_time = time.time()
    df = pl.read_csv(features_path)

    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("RF Test")

    # Start an MLflow run
    with mlflow.start_run():
        splits = np.arange(0.2, 0.3, 0.1)

        for split in splits:
            with mlflow.start_run(nested=True, run_name=f"Split_{split}"):
                X_train, X_test, y_train, y_test = train_test_split(df, train_size=split, gap=300)

                scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)

                params = {"n_estimators": 100, "max_depth": 128}

                model = random_forest_classifier(scaled_X_train, y_train.to_numpy().ravel(), **params)

                # save scaler and model to pickle file
                if save:
                    with open(model_path.with_stem(model_path.stem + "_scaler"), "wb") as f:
                        dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(model_path, "wb") as f:
                        dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.success(f"Model saved to: {model_path}")

                # evaluate model
                cm = metrics.confusion_matrix(y_test, model.predict(scaled_X_test), normalize="true")
                labels = df["ACTIVITY"].unique(maintain_order=True)
                cm = evaluate.confusion_matrix(model, labels, scaled_X_test, y_test)

                accuracy = metrics.accuracy_score(y_test, model.predict(scaled_X_test))
                logger.info("Accuracy: " + str(accuracy))
                logger.info("Score: " + str(model.score(scaled_X_test, y_test)))
                logger.info("Confusion Matrix:\n" + str(cm))

                # Log the hyperparameters
                params["split"] = split
                mlflow.log_params(params)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)

                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Training Info", "Basic RF model for labelled test data")

                # Infer the model signature
                signature = infer_signature(scaled_X_train, model.predict(scaled_X_train))

                # Log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="rf_model",
                    signature=signature,
                    input_example=scaled_X_train,
                )

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()

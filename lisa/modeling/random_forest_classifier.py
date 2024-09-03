import pickle
import time
from pathlib import Path
from pickle import dump

import polars as pl
import typer
from loguru import logger
from numpy import ndarray
from scipy.stats import randint
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from lisa import evaluate
from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR
from lisa.features import standard_scaler, train_test_split

app = typer.Typer()


def random_forest_classifier(X_train: ndarray, y_train: ndarray) -> RandomForestClassifier:
    """
    Fits a random forest classifier model to the input data.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels.

    Returns:
        RandomForestClassifier: The trained random forest classifier model.
    """
    rf = RandomForestClassifier(n_estimators=15, max_depth=128, n_jobs=-1)

    rf.fit(X_train, y_train)

    return rf

    param_dist = {"n_estimators": randint(50, 500), "max_depth": randint(1, 20)}
    rf = RandomForestClassifier()
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5, n_jobs=-1)
    rand_search.fit(X_train, y_train)

    return rand_search


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

    X_train, X_test, y_train, y_test = train_test_split(df, train_size=0.8, gap=300)

    scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)

    model = random_forest_classifier(scaled_X_train, y_train.to_numpy().ravel())
    # model = rand_search.best_estimator_

    # # Print the best hyperparameters
    # logger.info("Best hyperparameters:", rand_search.best_params_)

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

    logger.info("Accuracy: " + str(metrics.accuracy_score(y_test, model.predict(scaled_X_test))))
    logger.info("Score: " + str(model.score(scaled_X_test, y_test)))
    logger.info("Confusion Matrix:\n" + str(cm))

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    logger.info(f"Time taken to run: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    app()

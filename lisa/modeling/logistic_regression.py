import pickle
from pathlib import Path
from pickle import dump

import polars as pl
import typer
from loguru import logger
from numpy import ndarray
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR
from lisa.features import standard_scaler, train_test_split

app = typer.Typer()


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


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "pilot_data.csv",
    model_path: Path = MODELS_DIR / "logistic_regression.pkl",
    save: bool = typer.Option(False, help="Flag to save the model"),
):
    """
    Train a logistic regression classifier model on the processed data.
    Logs the score and confusion matrix, and optionally saves model to a pickle file.

    Args:
        features_path (Path): Path to the processed data.
        model_path (Path): Path to save the trained model.
        save (bool): Flag to save the model.
    """
    df = pl.read_csv(features_path)

    # TODO gap should be set my same variable as sliding window period in features.py
    X_train, X_test, y_train, y_test = train_test_split(df, train_size=0.8, gap=300)

    X_train, X_test, scaler = standard_scaler(X_train, X_test)

    model = logistic_regression(X_train, y_train)

    # save scaler and model to pickle file
    if save:
        with open(model_path.with_stem(model_path.stem + "_scaler"), "wb") as f:
            dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(model_path, "wb") as f:
            dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.success(f"Model saved to: {model_path}")

    # evaluate model
    cm = metrics.confusion_matrix(y_test, model.predict(X_test), normalize="true")

    logger.info("Score: " + str(model.score(X_test, y_test)))
    logger.info("Confusion Matrix:\n" + str(cm))


if __name__ == "__main__":
    app()

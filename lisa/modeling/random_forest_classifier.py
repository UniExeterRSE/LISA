import pickle
from pathlib import Path
from pickle import dump

import polars as pl
import typer
from loguru import logger
from numpy import ndarray
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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
    rf = RandomForestClassifier()

    rf.fit(X_train, y_train)

    return rf


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "pilot_data.csv",
    model_path: Path = MODELS_DIR / "random_forest.pkl",
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
    df = pl.read_csv(features_path)

    # TODO gap should be set my same variable as sliding window period in features.py
    X_train, X_test, y_train, y_test = train_test_split(df, train_size=0.8, gap=300)

    scaled_X_train, scaled_X_test, scaler = standard_scaler(X_train, X_test)

    model = random_forest_classifier(scaled_X_train, y_train)

    # save scaler and model to pickle file
    if save:
        with open(model_path.with_stem(model_path.stem + "_scaler"), "wb") as f:
            dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(model_path, "wb") as f:
            dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.success(f"Model saved to: {model_path}")

    # evaluate model
    cm = metrics.confusion_matrix(y_test, model.predict(scaled_X_test), normalize="true")

    logger.info("Accuracy: " + str(metrics.accuracy_score(y_test, model.predict(scaled_X_test))))
    logger.info("Score: " + str(model.score(scaled_X_test, y_test)))
    logger.info("Confusion Matrix:\n" + str(cm))

    feature_importances = pl.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    logger.info(feature_importances)


if __name__ == "__main__":
    app()

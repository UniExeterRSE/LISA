import pickle
from pathlib import Path
from pickle import dump

import polars as pl
import typer
from loguru import logger
from numpy import ndarray
from scipy.sparse import spmatrix
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def train_test_split(df: pl.DataFrame, train_size: float, gap: int = 0) -> list:
    """
    Splits the input dataframe into train and test sets.
    Each activity is split separately and sequentially in time, and then recombined.

    Args:
        df (pl.Dataframe): The input dataframe to be split.
        train_size (float): The proportion of rows to be included in the train set, between 0.0 and 1.0.
        gap (int, optional): The number of rows to leave as a gap between the train and test sets. Defaults to 0.

    Returns:
        list: A list containing train-test split of inputs, i.e. [X_train, X_test, y_train, y_test].
    """

    # Ensure train_size is between 0 and 1
    if not (0 <= train_size <= 1):
        raise ValueError(f"train_size must be between 0 and 1, but got {train_size}.")

    train_df = pl.DataFrame()
    test_df = pl.DataFrame()
    min_n_rows = float("inf")

    # Check if correct columns in df
    if "TRIAL" not in df.columns:
        logger.warning("TRIAL column not found in the dataframe.")
    if "TIME" not in df.columns:
        logger.warning("TIME column not found in the dataframe.")

    for activity in df["ACTIVITY"].unique(maintain_order=True):
        activity_df = df.filter(pl.col("ACTIVITY") == activity)

        n_rows = activity_df.height
        if n_rows < min_n_rows:
            min_n_rows = n_rows

        # Determine split indices
        train_split = int(train_size * n_rows)
        test_split = train_split + gap

        # Extract the first train_size% of rows
        activity_train_df = activity_df[:train_split]

        # Extract the next 1-train_size% of rows, leaving a gap of {gap} rows
        activity_test_df = activity_df[test_split:]

        train_df = train_df.vstack(activity_train_df)
        test_df = test_df.vstack(activity_test_df)

    # Check if gap is between 0 and min_n_rows
    if not (0 <= gap <= min_n_rows):
        raise ValueError(f"Gap must be between 0 and {min_n_rows}, but got {gap}.")

    return [
        train_df.select(pl.exclude(["ACTIVITY", "TRIAL", "TIME"])),
        test_df.select(pl.exclude(["ACTIVITY", "TRIAL", "TIME"])),
        train_df.select("ACTIVITY"),
        test_df.select("ACTIVITY"),
    ]


def standard_scaler(
    X_train: pl.DataFrame, X_test: pl.DataFrame
) -> tuple[ndarray, (ndarray | spmatrix), StandardScaler]:
    """
    Standardizes the input data.

    Args:
        X_train (pl.DataFrame): The training data to be standardised.
        X_test (pl.DataFrame): The test data to be standardised.

    Returns:
        tuple[ndarray, (ndarray | spmatrix), StandardScaler]: The standardised training and test data, and scaler.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler


def logistic_regression(X_train: ndarray, y_train: ndarray) -> OneVsRestClassifier:
    """
    Fits a logistic regression model to the input data.

    Args:
        X_train (ndarray): The training data.
        y_train (ndarray): The training labels.

    Returns:
        OneVsRestClassifier: The trained logistic regression model.
    """
    logisticRegr = OneVsRestClassifier(LogisticRegression())

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

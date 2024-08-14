from pathlib import Path

import polars as pl
import typer
from loguru import logger
from tqdm import tqdm

from lisa.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def train_test_split(df: pl.DataFrame, train_size: float, gap: int = 0) -> list:
    """
    Splits the input dataframe into train and test sets.

    Args:
        df (pl.Dataframe): The input dataframe to be split.
        train_size (float): The proportion of rows to be included in the train set, between 0.0 and 1.0.
        gap (int, optional): The number of rows to leave as a gap between the train and test sets. Defaults to 0.

    Returns:
        list: A list containing train-test split of inputs.
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


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

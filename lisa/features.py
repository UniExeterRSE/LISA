from pathlib import Path

import polars as pl
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler

from lisa.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def train_test_split(df: pl.DataFrame, train_size: float, gap: int = 0) -> list[pl.DataFrame]:
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


def standard_scaler(X_train: pl.DataFrame, X_test: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, StandardScaler]:
    """
    Standardizes the input data.

    Args:
        X_train (pl.DataFrame): The training data to be standardised.
        X_test (pl.DataFrame): The test data to be standardised.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, StandardScaler]: The standardised training and test data, and scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pl.from_numpy(X_train_scaled, schema=X_train.schema)
    X_test = pl.from_numpy(X_test_scaled, schema=X_test.schema)

    return X_train, X_test, scaler


def sliding_window(df: pl.DataFrame, period: int = 300, log: bool = False) -> pl.DataFrame:
    """
    Apply sliding window aggregation on a DataFrame.
    Extracts first, last, max, min, mean and std for each signal.

    Args:
        df (polars.DataFrame): The input DataFrame.
        period (int): The window size in number of rows. Default is 300.
        log (bool): Flag to enable logging. Default is False.

    Returns:
        polars.DataFrame: The aggregated DataFrame with rolling window statistics.

    """
    # List of categorical columns; one per trial
    categorical_columns = ["ACTIVITY", "SPEED", "INCLINE"]

    # List of other columns to exclude from aggregation
    exclude_columns = ["TIME", "TRIAL"]
    exclude_columns.extend(categorical_columns)

    # Get the list of columns to aggregate
    columns_to_aggregate = [col for col in df.columns if col not in exclude_columns]

    # Define the rolling window
    rolling = df.rolling(index_column="TIME", period=f"{period}i", group_by="TRIAL")

    # Define the aggregation operations
    aggregations = []
    for col in columns_to_aggregate:
        aggregations.extend(
            [
                pl.first(col).alias(f"first_{col}"),
                pl.last(col).alias(f"last_{col}"),
                pl.max(col).alias(f"max_{col}"),
                pl.min(col).alias(f"min_{col}"),
                pl.mean(col).alias(f"mean_{col}"),
                pl.std(col).alias(f"std_{col}"),
            ]
        )

    # Perform the aggregation
    if log:
        logger.info("Aggregating data...")
    result = rolling.agg(aggregations)

    # Check if TIME resets to 0 when TRIAL increases by 1
    trial_check = result.with_columns((pl.col("TRIAL") - pl.col("TRIAL").shift(1)).alias("TRIAL_INCREASE"))
    time_resets_correctly = trial_check.filter(pl.col("TRIAL_INCREASE") == 1)["TIME"].to_list() == [0] * len(
        trial_check.filter(pl.col("TRIAL_INCREASE") == 1)
    )

    # Remove rows before first 'full' window
    if time_resets_correctly:
        result = result.filter(pl.col("TIME") > period - 2)
    else:
        raise ValueError(
            "Time does not reset to 0 when TRIAL increases by 1. Unable to remove rows before first full window."
        )

    # Add the categorical columns back in by matching TRIAL
    def _add_columns_back(result, df, columns):
        for column in columns:
            column_map = dict(zip(df["TRIAL"], df[column], strict=True))
            result = result.with_columns(pl.col("TRIAL").replace_strict(column_map).alias(column))
        return result

    result = _add_columns_back(result, df, categorical_columns)

    return result


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "labelled_test_data.csv",
    save: bool = typer.Option(False, help="Flag to save the processed data to CSV"),
):
    """
    Run feature extraction on the interim data and save to CSV.
    Applies a 300ms sliding window to the data and calculates
    first, last, max, min, mean and std for each signal.

    Args:
        input_path (Path): Path to the directory containing the pilot data.
        output_path (Path): Path to save the processed data to.
        save (bool): Whether to save the processed data to a CSV file Defaults False.
    """
    df = pl.read_csv(input_path)

    df = sliding_window(df, period=300, log=True)

    if save:
        df.write_csv(output_path)
        logger.success(f"Output saved to: {output_path}")
    else:
        logger.success("Process complete, data not saved.")
        print(df.shape, df.describe(), df.head())


if __name__ == "__main__":
    app()

from pathlib import Path

import polars as pl
import typer
from loguru import logger

from lisa.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def sliding_window(df, period=300, log=False):
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
    # List of columns to exclude from aggregation
    exclude_columns = ["TIME", "TRIAL", "ACTIVITY"]

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
    trial_check = result.with_columns(
        (pl.col("TRIAL") - pl.col("TRIAL").shift(1)).alias("TRIAL_INCREASE")
    )
    time_resets_correctly = trial_check.filter(pl.col("TRIAL_INCREASE") == 1)[
        "TIME"
    ].to_list() == [0] * len(trial_check.filter(pl.col("TRIAL_INCREASE") == 1))

    # Remove rows before first 'full' window
    if time_resets_correctly:
        result = result.filter(pl.col("TIME") > period - 2)
    else:
        raise ValueError(
            "Time does not reset to 0 when TRIAL increases by 1. Unable to remove rows before first full window."
        )

    # Add the ACTIVITY column back in by matching TRIAL
    # Assumes every trial has one activity type
    activity_map = dict(zip(df["TRIAL"], df["ACTIVITY"]))
    result = result.with_columns(ACTIVITY=pl.col("TRIAL").replace_strict(activity_map))

    return result


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "pilot_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "pilot_data.csv",
    save: bool = typer.Option(False, help="Flag to save the processed data to CSV"),
):
    """
    Run feature extraction on the interim data and save to CSV.
    Applies a 300ms sliding window to the data and calculates
    first, last, max, min, mean and std for each signal.

    Args:
        input_path (Path): Path to the directory containing the pilot data.
        output_path (Path): Path to save the processed data to.
        save (bool): Whether to save the processed data to a CSV file.
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

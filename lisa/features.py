import json
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from lisa.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()


def sequential_stratified_split(
    lf: pl.LazyFrame,
    train_size: float,
    gap: int = 0,
    feature_cols: list[str] = ["ACTIVITY"],
) -> list[pl.LazyFrame]:
    """
    Splits the input LazyFrame into train and test sets.
    The data remains sequential (not shuffled), trials are not shared between train and test sets,
    and a gap of {gap} rows is left between the train and test sets.
    The split attempts to keep a balanced proportion of each feature in both sets.

    Args:
        lf (pl.LazyFrame): The input LazyFrame to be split.
        train_size (float): The proportion of rows to be included in the train set, between 0.0 and 1.0.
        gap (int, optional): The number of rows to leave as a gap between the train and test sets. Defaults to 0.
        feature_cols (list[str], optional): The list of feature columns to include in the split, to allow for multiple
            y features. Defaults to ['ACTIVITY'].

    Returns:
        list: A list containing train-test split of inputs, i.e. [X_train, X_test, y1_train, y1_test, y2_train, ...].
    """

    # Ensure train_size is between 0 and 1
    if not (0 <= train_size <= 1):
        raise ValueError(f"train_size must be between 0 and 1, but got {train_size}.")

    # Combine feature columns into a single column
    combined_feat_name = "_".join(feature_cols)
    if len(feature_cols) > 1:
        lf = lf.with_columns(
            pl.concat_str(
                [pl.col(col).fill_null("").cast(pl.Utf8) for col in feature_cols],
                separator="_",
            ).alias(combined_feat_name)
        )

    # Collect unique feature combinations lazily
    unique_features = lf.select(pl.col(combined_feat_name)).unique(maintain_order=True)

    def _process_feature(feature: str) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        feature_lf = lf.filter(pl.col(combined_feat_name) == feature)

        # Get number of rows for the feature group
        n_rows = feature_lf.select(pl.len()).collect().item()

        # Determine split indices
        train_split = int(train_size * n_rows)
        test_split = int(train_split + gap)

        # Get trial values lazily
        trial_values = feature_lf.select("TRIAL").collect().to_series()

        # Adjust train_split to avoid trial overlap
        for index in range(train_split, n_rows):
            if trial_values[index] != trial_values[train_split]:
                train_split = int(index)
                test_split = int(train_split + gap)
                break

        # Extract train and test splits lazily
        feature_train_lf = feature_lf.slice(0, train_split)
        feature_test_lf = feature_lf.slice(test_split, n_rows - test_split)

        return feature_train_lf, feature_test_lf

    train_lfs, test_lfs = [], []
    for feature in unique_features.collect().to_series():
        if feature is not None:
            train_lf, test_lf = _process_feature(feature)
            train_lfs.append(train_lf)
            test_lfs.append(test_lf)

    # Combine all train and test LazyFrames
    train_lf = pl.concat(train_lfs, rechunk=True)
    test_lf = pl.concat(test_lfs, rechunk=True)

    # Generate X and y splits lazily
    splits = [
        train_lf.select(
            pl.exclude(["ACTIVITY", "INCLINE", "SPEED", "TRIAL", "TIME", combined_feat_name] + feature_cols)
        ),
        test_lf.select(
            pl.exclude(["ACTIVITY", "INCLINE", "SPEED", "TRIAL", "TIME", combined_feat_name] + feature_cols)
        ),
    ]
    for feature in feature_cols:
        splits.extend([train_lf.select(feature), test_lf.select(feature)])

    return splits


def check_split_balance(
    y_test: pl.LazyFrame,
    y_train: pl.LazyFrame,
    threshold: float = 0.05,
) -> pl.DataFrame:
    """
    Helper function to check that the spread of values in the test and train sets are roughly similar.

    Args:
        y_test (pl.LazyFrame): The test set
        y_train (pl.LazyFrame): The train set
        tolerance (float, optional): The threshold for the difference between the two proportions, between 0 and 1.
            Defaults to 0.05.

    Returns:
        pl.DataFrame: A DataFrame containing the values that exceed the threshold, if any.
    """
    if y_test.collect_schema() != y_train.collect_schema():
        raise ValueError("The DataFrames must have the same schema")
    feature_name = y_test.collect_schema().names()[0]

    # Calculate the proportion of each value in the test and train sets
    train_proportions = (
        y_train.group_by(feature_name)
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion"))
        .drop("count")
    )

    test_proportions = (
        y_test.group_by(feature_name)
        .agg(pl.len().alias("count"))
        .with_columns((pl.col("count") / pl.col("count").sum()).alias("proportion_test"))
        .drop("count")
    )

    # Join the two LazyFrames on the common column
    df_merged = train_proportions.join(test_proportions, on=feature_name, how="inner", suffix="_test")

    # Calculate the absolute difference between the 'proportion' columns
    df_diff = df_merged.with_columns((pl.col("proportion") - pl.col("proportion_test")).abs().alias("diff"))

    # Filter rows where the difference exceeds the threshold
    return df_diff.filter(pl.col("diff") > threshold).collect()


def standard_scaler(X_train: pl.LazyFrame, X_test: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame, StandardScaler]:
    """
    Standardizes the input data.

    Args:
        X_train (pl.LazyFrame): The training data to be standardised.
        X_test (pl.LazyFrame): The test data to be standardised.

    Returns:
        tuple[pl.DataFrame, pl.DataFrame, StandardScaler]: The standardised training and test data, and scaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train.collect())
    X_test_scaled = scaler.transform(X_test.collect())

    X_train = pl.from_numpy(X_train_scaled, schema=X_train.collect_schema())
    X_test = pl.from_numpy(X_test_scaled, schema=X_test.collect_schema())

    return X_train, X_test, scaler


def sliding_window(
    df: pl.DataFrame,
    agg_columns: list[str],
    period: int,
    stats: list[str] = ["min", "max", "mean", "std"],
) -> pl.DataFrame:
    """
    Apply sliding window aggregation on a DataFrame.
    Extracts specified stats for each signal. Removes rows before the first full window.

    Args:
        df (pl.DataFrame): The input DataFrame.
        agg_columns (list[str]): The columns names to apply aggregation.
        period (int): The window size in number of rows.
        stats (list[str]): The statistics to calculate for each signal.
            Options are ['min', 'max', 'mean', 'std', 'first', 'last'].
            Default is ['min', 'max', 'mean', 'std'].
    Returns:
        pl.DataFrame: The processed DataFrame.
    """

    def _rolling_agg(
        chunk: pl.DataFrame,
        columns_to_aggregate: list[str],
        stats: list[str],
        period: int,
    ) -> pl.DataFrame:
        "Apply rolling aggregation to a single chunk."
        rolling = chunk.lazy().rolling(index_column="TIME", period=f"{period}i", group_by="TRIAL")

        # Map of statistic names to Polars functions
        stat_funcs = {
            "max": pl.max,
            "min": pl.min,
            "mean": pl.mean,
            "std": pl.std,
            "first": pl.first,
            "last": pl.last,
        }

        aggregations = []
        for col in columns_to_aggregate:
            for stat in stats:
                if stat in stat_funcs:
                    aggregations.append(stat_funcs[stat](col).alias(f"{stat}_{col}"))

        return rolling.agg(aggregations).collect()

    # Apply rolling aggregation
    result_chunk = _rolling_agg(df, agg_columns, stats, period)

    # Check if TIME resets to 0 when TRIAL increases by 1
    trial_check = result_chunk.with_columns((pl.col("TRIAL") - pl.col("TRIAL").shift(1)).alias("TRIAL_INCREASE"))
    time_resets_correctly = trial_check.filter(pl.col("TRIAL_INCREASE") == 1)["TIME"].to_list() == [0] * len(
        trial_check.filter(pl.col("TRIAL_INCREASE") == 1)
    )

    # Remove rows before first 'full' window
    if time_resets_correctly:
        result_chunk = result_chunk.filter(pl.col("TIME") > period - 2)
    else:
        raise ValueError(
            "Time does not reset to 0 when TRIAL increases by 1. " "Unable to remove rows before first full window."
        )

    return result_chunk


def feature_extraction(
    df: pl.DataFrame,
    output_path: Path,
    period: int = 300,
    stats: list[str] = ["min", "max", "mean", "std"],
    validate_schema: bool = True,
):
    """
    Apply sliding window aggregation, validates results and saves to Parquet file.

    Args:
        df (pl.DataFrame): The input DataFrame.
        output_path (Path): The output path to save the Parquet file.
        period (int): The window size in number of rows. Default is 300.
        stats (list[str]): The statistics to calculate for each signal.
            Options are ['min', 'max', 'mean', 'std', 'first', 'last']. Default is ['min', 'max', 'mean', 'std'].
        validate_schema (bool): Whether to validate the schema of the output DataFrame.
            Currently only works for 'full' dataset (i.e. all features). Default is True.
    """

    def _split_into_parts(df: pl.DataFrame, trials_per_part: int = 5) -> list[pl.DataFrame]:
        "Split df into groups of 'trials_per_part' TRIALs."

        # Get number of TRIALs
        max_trial = df["TRIAL"].max()

        # Split TRIALs into chunks of trials_per_part
        trial_chunks = [
            list(range(i, min(i + trials_per_part, max_trial + 1))) for i in range(0, max_trial + 1, trials_per_part)
        ]

        # Split the DataFrame based on the trial_chunks
        return [df.filter(pl.col("TRIAL").is_in(trial_chunk)) for trial_chunk in trial_chunks]

    # Load the schema for validation later
    if validate_schema:
        schema_path = Path(PROJ_ROOT / "lisa" / "validation_schema.json")
        with schema_path.open("r") as f:
            validation_schema = json.load(f)

    # List of categorical columns; one per trial
    categorical_columns = ["ACTIVITY", "SPEED", "INCLINE"]

    # List of other columns to exclude from aggregation
    exclude_columns = ["TIME", "TRIAL"]
    exclude_columns.extend(categorical_columns)

    # Get the list of columns to aggregate
    columns_to_aggregate = [col for col in df.collect_schema().names() if col not in exclude_columns]

    parts = _split_into_parts(df)

    for index, part in enumerate(tqdm(parts, desc="Processing Trial Groups")):
        result_chunk = sliding_window(part, columns_to_aggregate, period, stats)

        # Add the categorical columns back in by matching TRIAL
        def _add_columns_back(result, df, columns):
            for column in columns:
                column_map = dict(zip(df["TRIAL"], df[column], strict=True))
                result = result.with_columns(pl.col("TRIAL").replace_strict(column_map).alias(column))
            return result

        result_chunk = _add_columns_back(result_chunk, df, categorical_columns)

        # Validate the schema
        if validate_schema:
            result_schema = result_chunk.collect_schema()
            result_schema_dict = dict(
                zip(
                    result_schema.names(),
                    list(map(str, result_schema.dtypes())),
                    strict=True,
                )
            )
            diff = set(validation_schema.items()) ^ set(result_schema_dict.items())
            if diff:
                raise ValueError("Schema validation failed, difference: ", diff)

        # Convert DataFrame to PyArrow Table
        arrow_table = result_chunk.to_arrow()

        # Validate the Arrow table
        try:
            arrow_table.validate(full=True)
        except pa.lib.ArrowInvalid as e:
            logger.error(f"Arrow table validation failed: {e}")
            raise

        # Write the Arrow table to Parquet
        if index == 0:  # First chunk: initialize ParquetWriter
            writer = pq.ParquetWriter(output_path, arrow_table.schema)
        writer.write_table(arrow_table)

    writer.close()
    logger.success(f"All {len(parts)} parts processed and saved to {output_path}.")


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "main_data.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "main_data.parquet",
):
    """
    Run feature extraction on the interim data and save to file.
    Applies a 300ms sliding window to the data and calculates
    max, min, mean and std for each signal.

    Args:
        input_path (Path): Path to the directory containing the data from dataset.py.
        output_path (Path): Path to save the processed data to.
    """
    df = pl.read_parquet(input_path, low_memory=True, rechunk=True)

    feature_extraction(df, output_path, period=300, stats=["min", "max", "mean", "std"])


if __name__ == "__main__":
    app()

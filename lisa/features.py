from pathlib import Path

import polars as pl
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler

from lisa.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def sequential_stratified_split(
    df: pl.DataFrame,
    train_size: float,
    gap: int = 0,
    feature_cols: list[str] = ["ACTIVITY"],
) -> list[pl.DataFrame]:
    """
    Splits the input dataframe into train and test sets.
    The data remains sequential (not shuffled), trials are not shared between train and test sets,
    and a gap of {gap} rows is left between the train and test sets.
    The split attempts to keep a balanced proportion of each feature in both sets;
    this can be checked with check_split_balance().

    Args:
        df (pl.Dataframe): The input dataframe to be split.
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

    train_dfs = []
    test_dfs = []
    min_n_rows = float("inf")

    # Check if correct columns in df
    if "TRIAL" not in df.columns:
        logger.warning("TRIAL column not found in the dataframe.")
    if "TIME" not in df.columns:
        logger.warning("TIME column not found in the dataframe.")

    # Combine feature columns into a single column
    combined_feat_name = "_".join(feature_cols)
    df = df.with_columns(
        pl.concat_str(
            [pl.col(col).fill_null("").cast(pl.Utf8) for col in feature_cols],
            separator="_",
        ).alias(combined_feat_name)
    )
    # For each unique feature combination, split the data
    for feature in df[combined_feat_name].unique(maintain_order=True):
        feature_df = df.filter(pl.col(combined_feat_name) == feature)

        n_rows = feature_df.height
        if n_rows < min_n_rows:
            min_n_rows = n_rows

        # Determine split indices
        train_split = int(train_size * n_rows)
        test_split = train_split + gap

        # Adjust train_split to the closest index where 'TRIAL' changes,
        # to avoid same trial being in test and train sets
        trial_values = feature_df["TRIAL"].to_list()
        for index in range(train_split, n_rows):
            if trial_values[index] != trial_values[train_split]:
                train_split = index
                test_split = train_split + gap
                break
            # Account for edge case of last trial being below split threshold
            elif trial_values[index] == max(trial_values):
                unique_index = sorted(set(trial_values)).index(trial_values[index])
                lower_trial = sorted(set(trial_values))[unique_index - 1]
                train_split = max((i for i, x in enumerate(trial_values) if x == lower_trial))
                test_split = train_split + gap
                break

        # Extract the first train_size% of rows
        feature_train_df = feature_df[:train_split]

        # Extract the next 1-train_size% of rows, leaving a gap of {gap} rows
        feature_test_df = feature_df[test_split:]

        train_dfs.append(feature_train_df)
        test_dfs.append(feature_test_df)

    train_df = pl.concat(train_dfs, rechunk=True)
    test_df = pl.concat(test_dfs, rechunk=True)

    # Check if gap is between 0 and min_n_rows
    if not (0 <= gap <= min_n_rows):
        raise ValueError(f"Gap must be between 0 and {min_n_rows}, but got {gap}.")

    # Check if any trials are in both train and test sets
    common_trials = train_df["TRIAL"].value_counts().join(test_df["TRIAL"].value_counts(), on="TRIAL", how="inner")
    if not (common_trials.is_empty()):
        raise UserWarning(f"{common_trials.height} trials are in both train and test sets.")

    # Generate X data
    splits = [
        train_df.select(pl.exclude(["INCLINE", "SPEED", "TRIAL", "TIME", combined_feat_name] + feature_cols)),
        test_df.select(pl.exclude(["INCLINE", "SPEED", "TRIAL", "TIME", combined_feat_name] + feature_cols)),
    ]
    # Generate y data
    for feature in feature_cols:
        splits.extend([train_df.select(feature), test_df.select(feature)])

    return splits


def check_split_balance(y_test: pl.DataFrame, y_train: pl.DataFrame, threshold: float = 0.05) -> pl.DataFrame:
    """
    Helper function to check that the spread of values in the test and train sets are roughly similar.

    Args:
        y_test (pl.DataFrame): The test set
        y_train (pl.DataFrame): The train set
        tolerance (float, optional): The threshold for the difference between the two proportions, between 0 and 1.
            Defaults to 0.05.

    Returns:
        pl.DataFrame: A DataFrame containing the values that exceed the threshold, if any.
    """
    if y_test.columns != y_train.columns:
        raise ValueError("The DataFrames must have the same columns")
    feature_name = y_test.columns[0]

    # Calculate the proportion of each value in the test and train sets
    train_proportions = y_train.to_series().value_counts(sort=True, normalize=True)
    test_proportions = y_test.to_series().value_counts(sort=True, normalize=True)

    # Join the two DataFrames on the common column
    df_merged = train_proportions.join(test_proportions, on=feature_name, how="inner", suffix="_test")

    # Calculate the absolute difference between the 'proportion' columns
    df_diff = df_merged.with_columns((pl.col("proportion") - pl.col("proportion_test")).abs().alias("diff"))

    # Filter rows where the difference exceeds the threshold
    return df_diff.filter(pl.col("diff") > threshold)


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

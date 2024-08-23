import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from ezc3d import c3d
from loguru import logger
from tqdm import tqdm

from lisa.config import INTERIM_DATA_DIR, LABELLED_TEST_DATA_DIR

app = typer.Typer()


def _add_time_column(c: c3d, df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a time column to the given DataFrame.

    Args:
        c (c3d): The c3d object containing the data.
        df (pl.DataFrame): The DataFrame to add the time column to.

    Returns:
        pl.DataFrame: The DataFrame with the time column added.
    """
    frame_rate = c["parameters"]["ANALOG"]["RATE"]["value"][0]

    num_frames = c["data"]["analogs"].shape[2]

    time_data = np.arange(num_frames) / frame_rate

    # Convert time_data to milliseconds and cast to integers
    time_data_ms = (time_data * 1000).astype(int)
    time_series = pl.Series("TIME", time_data_ms)

    return df.with_columns(time_series)


def _find_column_names(c: c3d) -> list[str]:
    """
    Find column names in the given c3d object.
    Give numeric labels to any repeated columns (i.e. different IMU devices).

    Args:
        c (c3d): The c3d object containing the data.

    Returns:
        list[str]: A list of modified column names.

    """
    # TODO Currently we cannot map between the same device positions in different files.
    # Raw data will have to be updated.
    columns = c["parameters"]["ANALOG"]["LABELS"]["value"]

    # Count total occurrences of each item
    total_occurrences = {}
    for item in columns:
        if item in total_occurrences:
            total_occurrences[item] += 1
        else:
            total_occurrences[item] = 1

    modified_columns = []
    current_counts = {}

    for item in columns:
        # Only label items that appear more than once
        if total_occurrences[item] > 1:
            if item in current_counts:
                current_counts[item] += 1
            else:
                current_counts[item] = 1
            modified_item = f"D{current_counts[item]}_{item}"
        else:
            # If the item appears only once, keep it as is
            modified_item = item

        modified_columns.append(modified_item)

    return modified_columns


def _find_activity_category(filename: str, activity_categories: list[str]) -> str | None:
    """
    Find the activity category in the filename.

    Args:
        filename (str): The filename to search for the activity category.
        activity_categories (list[str]): A list of activity categories to search for.
    Returns:
        str | None: The activity category if found, otherwise None.
    """
    for activity in activity_categories:
        if activity in filename.lower():
            return activity
    return None  # Return None or a default value if no match is found


def _cartesian_to_spherical(df: pl.DataFrame, drop: bool = True) -> pl.DataFrame:
    """
    Convert features with cartesian coordinates to spherical coordinates.

    Args:
        df (pl.DataFrame): The DataFrame containing the cartesian coordinates.
        drop (bool): Whether to drop the original cartesian columns. Default True.

    Returns:
        pl.DataFrame: The DataFrame containing the spherical coordinates.
    """
    features = set([col.split(".")[0] for col in df.columns if ".x" in col])

    columns_to_drop = []

    # For each feature, compute spherical coordinates
    for feature in features:
        x_col = f"{feature}.x"
        y_col = f"{feature}.y"
        z_col = f"{feature}.z"

        if not all([x_col in df.columns, y_col in df.columns, z_col in df.columns]):
            logger.warning(f"Feature {feature} does not have all three cartesian coordinates. Skipping feature.")
            continue

        r_col = f"{feature}_r"
        df = df.with_columns((pl.col(x_col) ** 2 + pl.col(y_col) ** 2 + pl.col(z_col) ** 2).sqrt().alias(r_col))

        theta_col = f"{feature}_theta"
        df = df.with_columns(np.arccos(pl.col(z_col) / pl.col(r_col)).alias(theta_col))
        # Replace NaN values in theta_col with 0 for x=y=z=0 case
        df = df.with_columns(pl.col(theta_col).fill_nan(0).alias(theta_col))

        phi_col = f"{feature}_phi"
        df = df.with_columns(np.arctan2(pl.col(y_col), pl.col(x_col)).alias(phi_col))

        columns_to_drop.extend([x_col, y_col, z_col])

    if drop:
        df = df.drop(columns_to_drop)

    return df


@app.command()
def main(
    input_path: Path = LABELLED_TEST_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "labelled_test_data.csv",
    save: bool = typer.Option(False, help="Flag to save the processed data to CSV"),
    imu_label: bool = typer.Option(False, help="Flag if IMU data has location labels"),
):
    """
    Process pilot data and save to CSV.
    Removes unwanted columns, add an 'ACTIVITY' column based on the filename
    and combines all c3d files into one dataset.

    Args:
        input_path (Path): Path to the directory containing the pilot data.
        output_path (Path): Path to save the processed data to.
        save (bool): Whether to save the processed data to a CSV file. Default False.
        imu_label (bool): Flag if IMU data has location labels. Default False.
    """
    activity_categories = ["walk", "jog", "run", "jump"]
    total_df = None
    trial_count = 0

    for filename in tqdm(os.listdir(input_path)):
        # Ignore any non-c3d files, transition files or files that don't start with the activity categories,
        # i.e. calibration files
        if (
            filename.endswith(".c3d")
            and any(activity in filename.lower() for activity in activity_categories)
            and "transition" not in filename.lower()
        ):
            logger.info(f"Processing file: {filename}")
            file = os.path.join(input_path, filename)

            c3d_contents = c3d(file)

            analogs = c3d_contents["data"]["analogs"]
            df = pl.DataFrame(analogs[0].T)

            if df.is_empty():
                logger.warning(f"Skipping empty file: {filename}")
                continue

            df.columns = _find_column_names(c3d_contents)

            if imu_label:
                placement_labels = ["foot", "shank", "thigh", "pelvis"]
                filtered_columns = [
                    col for col in df.columns if any(label in col.lower() for label in placement_labels)
                ]
                df = df.select(filtered_columns)
            else:
                to_remove = [
                    "Force",
                    "Moment",
                    "Velocity",
                    "Angle.Pitch",
                    "Length.Sway",
                ]
                columns_to_remove = [col for col in df.columns if any(sub in col for sub in to_remove)]
                df = df.drop(columns_to_remove)

            # Add 'ACTIVITY', 'TIME' and 'TRIAL' columns
            df = df.with_columns(pl.lit(_find_activity_category(filename, activity_categories)).alias("ACTIVITY"))
            df = _add_time_column(c3d_contents, df)
            df = df.with_columns(pl.lit(trial_count).cast(pl.Int16).alias("TRIAL"))

            # df = _cartesian_to_spherical(df)
            trial_count += 1

            # Check for columns in df that are not in total_df
            if total_df is not None:
                df_columns = set(df.columns)
                total_df_columns = set(total_df.columns)
                extra_columns = df_columns - total_df_columns
                if extra_columns:
                    logger.warn(f"The following columns in df are not in total_df: {extra_columns}")
            total_df = df if total_df is None else total_df.vstack(df.select(total_df.columns))

    if save:
        total_df.write_csv(output_path)
        logger.success(f"Output saved to: {output_path}")
    else:
        print(total_df.describe(), total_df.head())
        logger.success("Process complete, data not saved.")


if __name__ == "__main__":
    app()

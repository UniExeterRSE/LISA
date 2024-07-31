import os
from pathlib import Path

import numpy as np
import polars as pl
import typer
from ezc3d import c3d
from loguru import logger
from tqdm import tqdm

from lisa.config import INTERIM_DATA_DIR, PILOT_DATA_DIR

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

    return df.with_columns(TIME=time_data)


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


@app.command()
def main(
    input_path: Path = PILOT_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "pilot_data.csv",
    save: bool = False,
):
    """
    Process pilot data and save to CSV.
    Removes unwanted columns, add an 'ACTIVITY' column based on the filename
    and combines all c3d files into one dataset.

    Args:
        input_path (Path): Path to the directory containing the pilot data.
        output_path (Path): Path to save the processed data to.
        save (bool): Whether to save the processed data to a CSV file.
    """
    activity_categories = ["walk", "jog", "run", "jump"]
    total_df = None

    for filename in tqdm(os.listdir(input_path)):

        # Ignore any non-c3d files or files that don't start with the activity categories, i.e. calibration files
        if filename.endswith(".c3d") and any(
            activity in filename.lower() for activity in activity_categories
        ):
            logger.info(f"Processing file: {filename}")
            file = os.path.join(PILOT_DATA_DIR, filename)

            c3d_contents = c3d(file)

            analogs = c3d_contents["data"]["analogs"]
            df = pl.DataFrame(analogs[0].T)

            if df.is_empty():
                logger.warning(f"Skipping empty file: {filename}")
                continue

            df.columns = _find_column_names(c3d_contents)

            to_remove = ["Force", "Moment", "Velocity", "Angle.Pitch", "Length.Sway"]

            columns_to_remove = [col for col in df.columns if any(sub in col for sub in to_remove)]

            df = df.drop(columns_to_remove)

            df = df.with_columns(
                pl.lit(_find_activity_category(filename, activity_categories)).alias("ACTIVITY")
            )

            df = _add_time_column(c3d_contents, df)

            if total_df is None:
                total_df = df
            else:
                total_df = total_df.vstack(df)

    if save:
        total_df.write_csv(output_path)
        logger.success(f"Output saved to: {output_path}")
    else:
        logger.success("Process complete, data not saved.")


if __name__ == "__main__":
    app()

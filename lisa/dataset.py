import os
from pathlib import Path

import polars as pl
import typer
from ezc3d import c3d
from loguru import logger
from tqdm import tqdm

from lisa.config import INTERIM_DATA_DIR, PILOT_DATA_DIR

app = typer.Typer()


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
):
    """
    Process pilot data and save to CSV.
    Removes unwanted columns, add an 'ACTIVITY' column based on the filename
    and combines all c3d files into one dataset.

    Args:
        input_path (Path): Path to the directory containing the pilot data.
        output_path (Path): Path to save the processed data to.
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

            if total_df is None:
                total_df = df
            else:
                total_df = total_df.vstack(df)

    total_df.write_csv(output_path)
    logger.success(f"Output saved to: {output_path}")


if __name__ == "__main__":
    app()

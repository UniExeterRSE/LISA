import os
import re
from pathlib import Path

import numpy as np
import polars as pl
from ezc3d import c3d
from loguru import logger
from tqdm import tqdm

from lisa.config import INTERIM_DATA_DIR, MAIN_DATA_DIR


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
    return None


def _find_incline(filename: str) -> int | None:
    """
    Find the incline in the filename. Decline is recorded as negative incline.
    Expects the format '2 incline' (space optional) for 2% incline.

    Args:
        filename (str): The filename to search for the incline.
    Returns:
        int | None: The incline if found, otherwise 0 for locomotion activities or None.
    """
    filename = filename.lower()

    # incline should default to None for jump
    if "jump" in filename:
        return None
    else:
        incline = 0
        if "incline" in filename:
            match = re.search(r"(\d+)(\s*|\_*)incline", filename)
            if match:
                incline = int(match.group(1))
        elif "decline" in filename:
            match = re.search(r"(\d+)(\s*|\_*)decline", filename)
            if match:
                incline = -int(match.group(1))

        return incline


def _find_speed(filename: str) -> float | None:
    """
    Find the speed in the filename. Expects the format '2_5ms' for 2.5 m/s.

    Args:
        filename (str): The filename to search for the speed.
    Returns:
        float | None: The speed if found, otherwise None.
    """
    speed = None
    if "ms" in filename.lower():
        match = re.search(r"(\d+)_(\d+)ms", filename)
        if match:
            speed = float(f"{match.group(1)}.{match.group(2)}")

    return speed


def process_c3d(
    c3d_contents: c3d,
    filename: str,
    activity_categories: list[str],
    trial_count: int,
    missing_label: str | None,
) -> pl.DataFrame | None:
    """
    Process a single c3d file and return a DataFrame.
    Unwanted columns are removed and 'ACTIVITY', 'INCLINE', 'SPEED', 'TIME' and 'TRIAL' columns are added.
    'jog' activities are relabelled to 'run'.

    Args:
        c3d_contents (c3d): The c3d object containing the data.
        filename (str): The filename of the c3d file.
        activity_categories (list[str]): A list of activity categories to search for.
        trial_count (int): The current trial count.
        missing_label (str | None): Body location label to use for any unlabelled data.

    Returns:
        pl.DataFrame | None: The processed data or None if no data found.
    """
    analogs = c3d_contents["data"]["analogs"]
    df = pl.DataFrame(analogs[0].T)

    if df.is_empty():
        return

    columns = c3d_contents["parameters"]["ANALOG"]["LABELS"]["value"]
    columns = [column.lower() for column in columns]

    if missing_label:
        measures = ["global angle", "highg", "accel", "gyro", "mag"]
        new_columns = []
        for col in columns:
            for measure in measures:
                if col.startswith(measure + "."):
                    # Add the location label
                    dimension = col[len(measure) :]  # Keep the dimension
                    col = f"{measure}_{missing_label}{dimension}"
            new_columns.append(col)
        columns = new_columns

    df.columns = columns

    placement_labels = ["foot", "shank", "thigh", "pelvis"]
    filtered_columns = [col for col in df.columns if any(label in col.lower() for label in placement_labels)]

    df = df.select(filtered_columns)

    # Add 'ACTIVITY', 'INCLINE', 'SPEED', 'TIME' and 'TRIAL' columns
    df = df.with_columns(pl.lit(_find_activity_category(filename, activity_categories)).alias("ACTIVITY"))
    # relabel jog to run
    df = df.with_columns(
        ACTIVITY=pl.when(pl.col("ACTIVITY") == "jog").then(pl.lit("run")).otherwise(pl.col("ACTIVITY"))
    )
    # Replace 'l_shank' with 'shank_l' in column names
    df = df.rename({col: col.replace("_l_shank", "_shank_l") for col in df.columns})

    # Replace any double underscores with singles in column names
    df = df.rename({col: col.replace("__", "_") for col in df.columns})

    df = df.with_columns(pl.lit(_find_incline(filename)).cast(pl.Int16).alias("INCLINE"))
    df = df.with_columns(pl.lit(_find_speed(filename)).cast(pl.Float32).alias("SPEED"))
    df = _add_time_column(c3d_contents, df)
    df = df.with_columns(pl.lit(trial_count).cast(pl.Int16).alias("TRIAL"))

    return df


def process_files(input_path: Path, missing_labels: dict, skip_participants: list) -> pl.LazyFrame:
    """
    Process c3d files in the given directory and return a single LazyFrame.

    Args:
        input_path (Path): Path to the directory containing the data.
        missing_labels (dict): If any body location labels are missing in the data, specify them here.
        skip_participants (list): Participant numbers to skip.

    Returns:
        pl.LazyFrame: The processed data.
    """
    # Activity verbs to search for in the filenames
    activity_categories = ["walk", "jog", "run", "jump"]

    total_df = None
    trial_count = 0

    # Process participants in order
    participants = sorted(os.listdir(input_path), key=lambda x: int(x.split("_")[0][1:]))

    for participant in tqdm(participants, desc="Participants"):
        participant_number = int(participant.split("_")[0][1:])

        # Skip certain participants
        if participant_number in skip_participants:
            logger.info(f"Skipping participant: {participant}")
            continue

        logger.info(f"Processing participant: {participant}")
        participant_path = os.path.join(input_path, participant)

        missing_label = missing_labels.get(participant_number)

        for filename in tqdm(os.listdir(participant_path), desc=f"Files in {participant}", leave=False):
            # Ignore any non-c3d files, transition files or files that don't start with the activity categories,
            # i.e. calibration files
            if (
                filename.endswith(".c3d")
                and any(activity in filename.lower() for activity in activity_categories)
                and "transition" not in filename.lower()
            ):
                # logger.info(f"Processing file: {filename}")
                file = os.path.join(participant_path, filename)

                c3d_contents = c3d(file)

                df = process_c3d(
                    c3d_contents,
                    filename,
                    activity_categories,
                    trial_count,
                    missing_label,
                )
                if df is None:
                    logger.warning(f"Skipping empty file: {filename}")
                    continue

                trial_count += 1

                # Check for columns in df that are not in total_df
                if total_df is not None:
                    df_columns = set(df.columns)
                    total_df_columns = set(total_df.columns)
                    extra_columns = df_columns - total_df_columns
                    if extra_columns:
                        logger.warning(f"The following columns in df are not in total_df: {extra_columns}")

                total_df = df if total_df is None else total_df.vstack(df.select(total_df.columns))
        logger.info(f"Processed participant: {participant}")

    return total_df.lazy()


def main(
    input_path: Path = MAIN_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR / "P1&P2.parquet",
    missing_labels: dict = {2: "thigh_l", 6: "pelvis", 7: "pelvis", 16: "thigh_l"},
    skip_participants: list = list(range(3, 17)),  # noqa: B008
):
    """
    Process pilot data and save to parquet.
    Combines all c3d files into one dataset.

    Args:
        input_path (Path): Path to the directory containing the input data.
        output_path (Path): Path to save the processed data to.
        missing_labels (dict): If any body location labels are missing in the data, specify them here.
        skip_participants (list): Participant numbers to skip.
    """

    data = process_files(input_path, missing_labels, skip_participants)

    data.sink_parquet(output_path)
    logger.success(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

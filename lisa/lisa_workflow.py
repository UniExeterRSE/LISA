from pathlib import Path

from loguru import logger
from tqdm import tqdm

from lisa.config import MAIN_DATA_DIR, PROCESSED_DATA_DIR
from lisa.dataset import process_files
from lisa.features import feature_extraction
from lisa.modeling.multipredictor import multipredictor


def main(
    input_path: Path = MAIN_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "example_file.parquet",
    models: list[str] = ["LR", "RF", "LGBM"],
    run_id: str = "fix",
    missing_labels: dict[int, str] = {
        2: "thigh_l",
        6: "pelvis",
        7: "pelvis",
        16: "thigh_l",
    },
    skip_participants: list[int] = [15, 16],
    window: int = 800,
    split: float = 0.8,
    measures=["global angle", "mag", "gyro", "accel"],
    locations=["pelvis", "thigh", "shank", "foot_", "foot sensor"],
    dimensions=["z"],
    stats=["min", "max"],
):
    """
    Top-level script for the end-to-end processing of the LISA dataset.

    Args:
        input_path (Path): Path to the raw data directory. Defaults to the main data directory.
        output_path (Path): File path to save the processed data to. Defaults to 'example_file.parquet'.
        models (list[str]): Model 'families' to train.
                    Currently supports 'LR' (logistic/linear regression), 'RF' (random forest), 'LGBM' (LightGBM).
                    Defaults to all three.
        run_id (str): Unique identifier for the run. Defaults to an empty string.
        missing_labels (dict[int, str]): IMU location labels known to be missing in the data.
                    Defaults to {2: 'thigh_l', 6: 'pelvis', 7: 'pelvis', 16: 'thigh_l'}.
        skip_participants (list[int]): List of participant IDs to skip (i.e. for separate test set).
                    Defaults to [15, 16].
        window (int): Window size for feature extraction. Defaults to 800.
        split (float): Train-test split ratio. Defaults to 0.8.
        measures (list[str]): Measures to extract. Defaults to ['global angle', 'mag', 'gyro', 'accel'].
        locations (list[str]): Locations to extract. Defaults to ['pelvis', 'thigh', 'shank', 'foot_', 'foot sensor'].
        dimensions (list[str]): Dimensions to extract. Defaults to ['z'].
        stats (list[str]): Statistics to calculate. Defaults to ['min', 'max'].
    """

    feature_extraction(
        process_files(
            input_path,
            skip_participants,
            missing_labels,
            measures,
            locations,
            dimensions,
        ).collect(),
        output_path,
        window,
        stats,
        False,
    )
    logger.info("Completed processing")

    for model in tqdm(models):
        run_name = model + "_" + run_id
        multipredictor(output_path, run_name, model, window, split, True)

    logger.success("Completed training")


if __name__ == "__main__":
    main()

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from lisa.config import MAIN_DATA_DIR, PROCESSED_DATA_DIR
from lisa.dataset import process_files
from lisa.features import feature_extraction
from lisa.modeling.remote_multipredictor import main as multipredictor_main


def main(
    input_path: Path = MAIN_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "reduced_main_data.parquet",
    models: list[str] = ["LGBM", "RF", "LR"],
    run_id: str = "z",
    measures=["global angle", "mag", "gyro", "accel"],
    locations=["pelvis", "thigh", "shank", "foot_", "foot sensor"],
    dimensions=["z"],
    stats=["min", "max"],
):
    """
    Script for end-to-end processing of the LISA dataset.
    """

    missing_labels = {2: "thigh_l", 6: "pelvis", 7: "pelvis", 16: "thigh_l"}
    skip_participants = list(range(1, 17))
    skip_participants = [15, 16]

    window = 800
    split = 0.8

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
        run_id = model + "_" + run_id
        multipredictor_main(output_path, run_id, model, window, split, True)

    logger.success("Completed training")


if __name__ == "__main__":
    main(run_id="yz", dimensions=["y", "z"])
    main(run_id="mean", stats=["mean", "min", "max"])

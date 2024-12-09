from pathlib import Path

from loguru import logger

from lisa.config import MAIN_DATA_DIR, PROCESSED_DATA_DIR
from lisa.dataset import process_files
from lisa.features import sliding_window
from lisa.modeling.multipredictor import main as multipredictor_main


def main(
    input_path: Path = MAIN_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "P1.parquet",
    model: str = "LGBM",
    run_id: str = "P1",
):
    """
    Script for end-to-end processing of the LISA dataset.
    """

    missing_labels = {2: "thigh_l", 6: "pelvis", 7: "pelvis", 16: "thigh_l"}
    skip_participants = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    window = 300
    split = 0.8

    logger.info("Completed data")
    sliding_window(
        process_files(input_path, missing_labels, skip_participants).collect(),
        output_path,
        window,
    )
    logger.info("Completed processing")
    multipredictor_main(output_path, run_id, model, window, split)
    logger.success("Completed training")


if __name__ == "__main__":
    main()

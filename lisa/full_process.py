from pathlib import Path

import polars as pl
from loguru import logger

from lisa.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from lisa.features import feature_extraction
from lisa.modeling.remote_multipredictor import main as multipredictor_main


def main(
    input_path: Path = INTERIM_DATA_DIR / "main_data.parquet",
    output_path: Path = PROCESSED_DATA_DIR / "main_main_data.parquet",
    model: str = "LGBM",
    run_id: str = "remote_test",
):
    """
    Script for remote processing of the LISA dataset.
    """

    window = 800
    split = 0.8
    stats = ["min", "max"]

    logger.info("Starting processing")

    feature_extraction(
        pl.read_parquet(input_path),
        output_path,
        window,
        stats,
        False,
    )
    logger.info("Completed processing")
    multipredictor_main(output_path, run_id, model, window, split)
    logger.success("Completed training")


if __name__ == "__main__":
    main()

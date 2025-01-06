from pathlib import Path

import polars as pl

from lisa.config import PROCESSED_DATA_DIR


def main(data_path: Path = PROCESSED_DATA_DIR / "main_main_data.parquet"):
    lf = pl.scan_parquet(data_path)

    lf.collect()


if __name__ == "__main__":
    main()

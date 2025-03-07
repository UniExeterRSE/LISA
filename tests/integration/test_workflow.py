import shutil
from pathlib import Path

from lisa.config import MODELS_DIR
from lisa.dataset import create_synthetic_c3d_file, process_files
from lisa.features import feature_extraction
from lisa.modeling.multipredictor import multipredictor


def test_integration():
    """
    Test the full LISA workflow:
        - Process C3D files
        - Extract features
        - Train and test models

    The test does not assert any results, but checks if the workflow runs without errors.
    Adapted from the example notebook.
    """
    # Create directories for test data
    test_file_dir = Path(__file__).parent
    data_dir = test_file_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    p1_dir = data_dir / "P1"
    p2_dir = data_dir / "P2"
    p1_dir.mkdir(exist_ok=True)
    p2_dir.mkdir(exist_ok=True)

    # Create synthetic C3D files
    create_synthetic_c3d_file(p1_dir / "P1_Walk_1_7ms_10Incline.c3d")
    create_synthetic_c3d_file(p2_dir / "P2_Run_3_0ms_5Decline.c3d")

    # Process the synthetic C3D files
    processed_data = process_files(data_dir)
    processed_data = processed_data.collect()

    # Create directories for processed data
    processed_data_dir = Path("processed_data")
    processed_data_dir.mkdir(exist_ok=True)
    features_path = processed_data_dir / "features.parquet"

    # Extract features
    window_size = 1000
    feature_extraction(processed_data, features_path, period=window_size, validate_schema=False)

    # Train and test models
    run_name = "integration_test"

    multipredictor(
        features_path,
        run_name=run_name,
        model="LGBM",
        window=window_size,
        split=0.8,
        save=False,
    )

    # Clean up created directories and files
    shutil.rmtree(data_dir)
    shutil.rmtree(processed_data_dir)
    shutil.rmtree(MODELS_DIR / run_name)


if __name__ == "__main__":
    test_integration()

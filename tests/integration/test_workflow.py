import json
import shutil
from pathlib import Path

import pytest

from lisa.config import MODELS_DIR, PROJ_ROOT
from lisa.dataset import create_synthetic_c3d_file, process_files
from lisa.features import feature_extraction
from lisa.modeling.multipredictor import multipredictor


def test_integration():
    """
    Test the full LISA workflow:
        - Process C3D files
        - Extract features
        - Train and test models

    The test checks if the workflow runs without errors and if the output is as expected.
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
    create_synthetic_c3d_file(p1_dir / "P1_Walk_1_7ms_10Incline.c3d", 42)
    create_synthetic_c3d_file(p2_dir / "P2_Run_3_0ms_5Decline.c3d", 43)

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

    # assert output is correct
    with open(MODELS_DIR / run_name / "output.json") as f:
        output = json.load(f)

    expected_scores = {
        "activity": 0.4794007490636704,
        "activity_weighted": 0.47885165971057453,
        "speed_r2": -0.48746998789033325,
        "speed_rmse": 0.7927521830814496,
        "incline_r2": -0.6304012893127671,
        "incline_rmse": 9.576537606245962,
    }

    for key, expected_value in expected_scores.items():
        assert pytest.approx(output["score"][key], rel=1e-6) == expected_value, f"{key} does not match"

    # assert hyperparameters are correct
    with open(PROJ_ROOT / "lisa" / "modeling" / "hyperparameters.json") as f:
        expected_hyperparams = json.load(f)
    assert output["params"]["hyperparams"] == expected_hyperparams["LGBM"]

    # assert all other files are created
    assert (MODELS_DIR / run_name / "confusion_matrix.png").exists()
    assert (MODELS_DIR / run_name / "Speed_hist.png").exists()
    assert (MODELS_DIR / run_name / "feature_importances_Speed.json").exists()
    assert (MODELS_DIR / run_name / "Incline_hist.png").exists()
    assert (MODELS_DIR / run_name / "feature_importances_Incline.json").exists()

    # Clean up created directories and files
    shutil.rmtree(data_dir)
    shutil.rmtree(processed_data_dir)
    shutil.rmtree(MODELS_DIR / run_name)


if __name__ == "__main__":
    test_integration()
    test_integration()
    test_integration()

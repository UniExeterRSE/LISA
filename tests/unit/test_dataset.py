import numpy as np
import polars as pl
from ezc3d import c3d
from polars.testing import assert_frame_equal

from lisa.dataset import process_c3d


def test_process_c3d() -> None:
    """
    Test process_c3d function.
    Mock c3d object has structure:
    {
        "data": {"analogs": [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]},
        "parameters": {
            "ANALOG": {
                "RATE": {"value": [100]},
                "LABELS": {"value": ["Global Angle_Foot_L.x", "Global Angle_Foot_L.y", "Global Angle_Foot_L.z"]},
            }
        },
    }
    """
    # Create a mock c3d object
    c3d_contents = c3d()
    c3d_contents["data"]["analogs"] = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    c3d_contents["parameters"]["ANALOG"]["RATE"]["value"] = np.array([100])
    c3d_contents["parameters"]["ANALOG"]["LABELS"]["value"] = [
        "Global Angle_Foot_L.x",
        "Global Angle_Foot_L.y",
        "Global Angle_Foot_L.z",
    ]

    filename = "Jogging2_5ms_weighted_10 decline"
    activity_categories = ["walk", "jog", "run", "jump"]
    imu_label_exists = True
    trial_count = 0

    # Call the process_c3d function
    result = process_c3d(c3d_contents, filename, activity_categories, imu_label_exists, trial_count)

    # Check the result
    expected_result = pl.DataFrame(
        {
            "Global Angle_Foot_L.x": [1, 2, 3],
            "Global Angle_Foot_L.y": [4, 5, 6],
            "Global Angle_Foot_L.z": [7, 8, 9],
            "ACTIVITY": ["run", "run", "run"],
            "INCLINE": [-10, -10, -10],
            "SPEED": [2.5, 2.5, 2.5],
            "TIME": [0, 10, 20],
            "TRIAL": [0, 0, 0],
        }
    )

    assert_frame_equal(
        result,
        expected_result,
        check_column_order=False,
        check_dtypes=False,
        rtol=1e-3,
    )


def test_process_c3d_empty() -> None:
    """
    Test process_c3d function with empty data
    """
    # Create a mock c3d object with empty data
    c3d_contents = c3d()

    filename = "walk_2_incline_3_5ms.c3d"
    activity_categories = ["walk", "jog", "run", "jump"]
    imu_label_exists = False
    trial_count = 1

    # Call the process_c3d function
    result = process_c3d(c3d_contents, filename, activity_categories, imu_label_exists, trial_count)

    # Check the result
    assert result is None

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lisa.features import sliding_window


def test_sliding_window_single_trial() -> None:
    """
    Test sliding_window with a single trial
    """
    # Create a sample DataFrame
    df = pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "TIME": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "ACTIVITY": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
            "Value": [10, 20, 30, 40, 50, 60, 70, 100, 90, 80],
        },
        strict=False,
    )

    # Call the sliding_window function
    result = sliding_window(df, period=3)

    # Check the result
    expected_result = pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 0, 0, 0, 0],
            "TIME": [2, 3, 4, 5, 6, 7, 8, 9],
            "ACTIVITY": ["A", "A", "A", "A", "A", "A", "A", "A"],
            "first_Value": [10, 20, 30, 40, 50, 60, 70, 100],
            "last_Value": [30, 40, 50, 60, 70, 100, 90, 80],
            "max_Value": [30, 40, 50, 60, 70, 100, 100, 100],
            "min_Value": [10, 20, 30, 40, 50, 60, 70, 80],
            "mean_Value": [20, 30, 40, 50, 60, 76.66667, 86.66667, 90],
            "std_Value": [10, 10, 10, 10, 10, 20.81666, 15.27525, 10],
        },
        strict=False,
    )

    assert_frame_equal(result, expected_result, check_column_order=False, check_dtypes=False)


def test_sliding_window_multi_trial() -> None:
    """
    Test sliding_window with multiple trials
    """
    # Create a sample DataFrame
    df = pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 3, 0, 1, 2, 0, 1, 2],
            "ACTIVITY": ["A", "A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )

    # Call the sliding_window function
    result = sliding_window(df, period=3)

    # Check the result
    expected_result = pl.DataFrame(
        {
            "TRIAL": [0, 0, 1, 2],
            "TIME": [2, 3, 2, 2],
            "ACTIVITY": ["A", "A", "B", "A"],
            "first_Value": [10, 20, 50, 80],
            "last_Value": [30, 40, 70, 100],
            "max_Value": [30, 40, 70, 100],
            "min_Value": [10, 20, 50, 80],
            "mean_Value": [20, 30, 60, 90],
            "std_Value": [10, 10, 10, 10],
        },
        strict=False,
    )

    assert_frame_equal(result, expected_result, check_column_order=False, check_dtypes=False)


def test_sliding_window_time_reset_error() -> None:
    """
    Test sliding_window raises an error if time does not reset for new trial
    """
    # Create a sample DataFrame
    df = pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            "ACTIVITY": ["A", "A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )

    with pytest.raises(ValueError):
        result = sliding_window(df, period=3)

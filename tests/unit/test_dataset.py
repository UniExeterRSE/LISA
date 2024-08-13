import polars as pl
from polars.testing import assert_frame_equal

from lisa.dataset import _cartesian_to_spherical


def test_cartesian_to_spherical() -> None:
    """
    Test _cartesian_to_spherical function
    """
    # Create a sample DataFrame with cartesian coordinates
    df = pl.DataFrame(
        {
            "feature.x": [1, 2, 3],
            "feature.y": [4, 5, 6],
            "feature.z": [7, 8, 9],
        }
    )

    # Call the _cartesian_to_spherical function
    result = _cartesian_to_spherical(df)

    # Check the result
    expected_result = pl.DataFrame(
        {
            "feature_r": [8.124, 9.644, 11.225],
            "feature_theta": [0.5323, 0.5925, 0.6405],
            "feature_phi": [1.326, 1.19, 1.107],
        }
    )

    assert_frame_equal(result, expected_result, check_column_order=False, check_dtypes=False, rtol=1e-3)


def test_cartesian_to_spherical_zeros() -> None:
    """
    Test _cartesian_to_spherical function when all values are zero
    """
    # Create a sample DataFrame with cartesian coordinates
    df = pl.DataFrame(
        {
            "feature.x": [0],
            "feature.y": [0],
            "feature.z": [0],
        }
    )

    # Call the _cartesian_to_spherical function
    result = _cartesian_to_spherical(df)

    # Check the result
    expected_result = pl.DataFrame(
        {
            "feature_r": [0],
            "feature_theta": [0],
            "feature_phi": [0],
        }
    )

    assert_frame_equal(result, expected_result, check_column_order=False, check_dtypes=False, rtol=1e-3)


def test_cartesian_to_spherical_missing_dim():
    """
    Test conversion skips features with missing dimensions
    """

    # Create a sample DataFrame with missing z coordinate for feature 'A'
    df = pl.DataFrame(
        {
            "A.x": [1, 2, 3],
            "A.y": [4, 5, 6],
            "B.x": [7, 8, 9],
            "B.y": [10, 11, 12],
            "B.z": [13, 14, 15],
        }
    )

    # Call the _cartesian_to_spherical function
    result = _cartesian_to_spherical(df)

    # Check the result
    expected_headings = ["A.x", "A.y", "B_r", "B_theta", "B_phi"]

    assert result.columns == expected_headings

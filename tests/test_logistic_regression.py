import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lisa.modeling.logistic_regression import train_test_split


@pytest.fixture
def sample_dataframe():
    return pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            "ACTIVITY": ["A", "A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )


def test_train_test_split(sample_dataframe):
    """
    Test train_test_split function
    """

    # Call the train_test_split function
    train_data, test_data, train_labels, test_labels = train_test_split(sample_dataframe, train_size=0.8, gap=0)

    # Check the train_data result
    expected_train_data = pl.DataFrame(
        {
            "Value": [10, 20, 30, 40, 80, 50, 60],
        },
        strict=False,
    )
    assert_frame_equal(train_data, expected_train_data, check_column_order=False, check_dtypes=False)

    # Check the test_data result
    expected_test_data = pl.DataFrame(
        {
            "Value": [90, 100, 70],
        },
        strict=False,
    )
    assert_frame_equal(test_data, expected_test_data, check_column_order=False, check_dtypes=False)

    # Check the train_labels result
    expected_train_labels = pl.DataFrame(
        {
            "ACTIVITY": ["A", "A", "A", "A", "A", "B", "B"],
        },
        strict=False,
    )
    assert_frame_equal(
        train_labels,
        expected_train_labels,
        check_column_order=False,
        check_dtypes=False,
    )

    # Check the test_labels result
    expected_test_labels = pl.DataFrame(
        {
            "ACTIVITY": ["A", "A", "B"],
        },
        strict=False,
    )
    assert_frame_equal(test_labels, expected_test_labels, check_column_order=False, check_dtypes=False)
    assert_frame_equal(test_labels, expected_test_labels, check_column_order=False, check_dtypes=False)


def test_train_test_split_gap(sample_dataframe):
    """
    Test train_test_split gap parameter
    """

    # Check error not raised when 0 <= gap <= min_n_rows
    train_test_split(sample_dataframe, train_size=0.8, gap=2)

    # Check error is raised when gap < 0
    with pytest.raises(ValueError):
        train_test_split(sample_dataframe, train_size=0.8, gap=-5)

    # Check error is raised when gap > min_n_rows
    with pytest.raises(ValueError):
        train_test_split(sample_dataframe, train_size=0.8, gap=4)


def test_train_test_split_train_size(sample_dataframe):
    """
    Test train_test_split train_size parameter
    """

    # Check error is raised when train_size < 0
    with pytest.raises(ValueError):
        train_test_split(sample_dataframe, train_size=-0.8)

    # Check error is raised when train_size > 1
    with pytest.raises(ValueError):
        train_test_split(sample_dataframe, train_size=1.8)

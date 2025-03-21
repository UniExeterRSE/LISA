import polars as pl
import pytest
from polars.testing import assert_frame_equal

from lisa.features import (
    check_split_balance,
    sequential_stratified_split,
    sliding_window,
)


@pytest.fixture
def sample_lazyframe():
    return pl.LazyFrame(
        {
            "TRIAL": [0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
            "TIME": [0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            "ACTIVITY": ["A", "A", "A", "A", "B", "B", "B", "A", "A", "A"],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )


def test_sequential_stratified_split(sample_lazyframe):
    """
    Test sequential_stratified_split function
    """

    # Call the train_test_split function
    train_data, test_data, train_labels, test_labels = sequential_stratified_split(
        sample_lazyframe, train_size=0.8, gap=0
    )

    # Check the train_data result
    expected_train_data = pl.LazyFrame(
        {
            "Value": [10, 20, 30, 40, 80, 50, 60],
        },
        strict=False,
    )
    assert_frame_equal(train_data, expected_train_data, check_column_order=False, check_dtypes=False)

    # Check the test_data result
    expected_test_data = pl.LazyFrame(
        {
            "Value": [90, 100, 70],
        },
        strict=False,
    )
    assert_frame_equal(test_data, expected_test_data, check_column_order=False, check_dtypes=False)

    # Check the train_labels result
    expected_train_labels = pl.LazyFrame(
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
    expected_test_labels = pl.LazyFrame(
        {
            "ACTIVITY": ["A", "A", "B"],
        },
        strict=False,
    )
    assert_frame_equal(test_labels, expected_test_labels, check_column_order=False, check_dtypes=False)


def test_sequential_stratified_split_gap(sample_lazyframe):
    """
    Test sequential_stratified_split gap parameter
    """

    # Check error is raised when gap bigger than trials
    with pytest.raises(ValueError):
        sequential_stratified_split(sample_lazyframe, train_size=0.8, gap=4)


def test_sequential_stratified_split_train_size(sample_lazyframe):
    """
    Test sequential_stratified_split train_size parameter
    """

    # Check error is raised when train_size < 0
    with pytest.raises(ValueError):
        sequential_stratified_split(sample_lazyframe, train_size=-0.8)

    # Check error is raised when train_size > 1
    with pytest.raises(ValueError):
        sequential_stratified_split(sample_lazyframe, train_size=1.8)


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
            "SPEED": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "INCLINE": [None, None, None, None, None, None, None, None, None, None],
            "Value": [10, 20, 30, 40, 50, 60, 70, 100, 90, 80],
        },
        strict=False,
    )

    # Call the sliding_window function
    result = sliding_window(df, ["Value"], 3, stats=["max", "min", "mean", "std"])

    # Check the result
    expected_result = pl.DataFrame(
        {
            "TRIAL": [0, 0, 0, 0, 0, 0, 0, 0],
            "TIME": [2, 3, 4, 5, 6, 7, 8, 9],
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
            "SPEED": [2.3, 2.3, 2.3, 2.3, None, None, None, 3.4, 3.4, 3.4],
            "INCLINE": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )

    # Call the sliding_window function
    result = sliding_window(df, ["Value"], 3, stats=["max", "min", "mean", "std"])

    # Check the result
    expected_result = pl.DataFrame(
        {
            "TRIAL": [0, 0, 1, 2],
            "TIME": [2, 3, 2, 2],
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
            "SPEED": [2.3, 2.3, 2.3, 2.3, None, None, None, 3.4, 3.4, 3.4],
            "INCLINE": [-1, -1, -1, -1, 0, 0, 0, 1, 1, 1],
            "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        },
        strict=False,
    )

    with pytest.raises(ValueError):
        sliding_window(df, ["Value"], 3)


def test_check_split_balance():
    """
    Test check_split_balance function
    """
    # Create sample data
    df1 = pl.LazyFrame({"ACTIVITY": ["walk", "walk", "run", "run"]})
    df2 = pl.LazyFrame({"ACTIVITY": ["walk", "walk", "walk", "run", "run", "run"]})

    assert check_split_balance(df1, df2).is_empty()

    df3 = pl.LazyFrame({"ACTIVITY": ["walk", "walk", "walk", "run"]})

    expected_difference = pl.DataFrame(
        {
            "ACTIVITY": ["run", "walk"],
            "proportion": [0.25, 0.75],
            "proportion_test": [0.5, 0.5],
            "diff": [0.25, 0.25],
        }
    )

    assert_frame_equal(
        check_split_balance(df1, df3),
        expected_difference,
        check_row_order=False,
    )

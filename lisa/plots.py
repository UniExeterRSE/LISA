from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn import metrics
from sklearn.base import BaseEstimator


def activity_weight_pie_chart(df: pl.DataFrame) -> plt.Figure:
    """
    Create a pie chart of the proportional occurrence of activities in the dataset.

    Args:
        df (pl.DataFrame): The DataFrame containing the 'ACTIVITY' column.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    activity_counts = df["ACTIVITY"].value_counts()
    df_dict = dict(
        zip(
            activity_counts["ACTIVITY"].to_list(),
            activity_counts["count"].to_list(),
            strict=True,
        )
    )

    # Format plot
    labels = list(label.capitalize() for label in df_dict)
    plt.rcParams.update({"font.size": 32})
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("Set2")

    # Create pie chart
    return plt.pie(
        df_dict.values(),
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        shadow=True,
    )


def confusion_matrix_plot(
    cm: np.ndarray,
    model: BaseEstimator,
    labels: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
) -> plt.Figure:
    """
    Plot a confusion matrix from a confusion matrix ndarray.

    Args:
        cm (np.ndarray): The confusion matrix.
        model (BaseEstimator): The trained model.
        labels (pl.Series): The category labels.
        X_test (pl.DataFrame): The test features.
        y_test (pl.DataFrame): The test labels.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    plt.rcParams.update({"font.size": 16})
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.str.to_titlecase())
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues_r", values_format=".2%", colorbar=False)
    # all_sample_title = f"Score: {str(model.score(X_test, y_test))}"
    # ax.set_title(all_sample_title, size=15)
    plt.tight_layout()

    return fig


def regression_histogram(
    y_true: pl.DataFrame,
    y_pred: np.ndarray,
    y_name: Literal["SPEED", "INCLINE"],
) -> plt.Figure:
    """
    Plot a histogram of the predicted values versus the true values after regression.
    Predicted values are displayed twice; once binned to match the true values, and once in a finer distribution.

    Args:
        y_true (pl.DataFrame): The true values.
        y_pred (np.ndarray): The predicted values.
        y_name (Literal["SPEED", "INCLINE"]): The name of the target column.

    Returns:
        fig: The matplotlib figure object.
    """

    if y_name not in y_true.columns:
        y_name = y_name.upper()

    counts = y_true[y_name].value_counts(sort=True)

    # Find the bin edges, taking the 'true' data as midpoints
    midpoints = counts[y_name].sort()
    bin_edges = [
        midpoints[0] - (midpoints[1] - midpoints[0]) / 2,
        *[(midpoints[i] + midpoints[i + 1]) / 2 for i in range(len(midpoints) - 1)],
        midpoints[-1] + (midpoints[-1] - midpoints[-2]) / 2,
    ]
    bar_width = (bin_edges[2] - bin_edges[1]) * 0.2
    fig, ax = plt.subplots()

    ax.hist(y_pred, bins=bin_edges, alpha=0.6, label="Binned Predicted Value")
    ax.bar(
        counts[y_name],
        counts["count"],
        color="red",
        alpha=0.6,
        label="Actual Value",
        width=bar_width,
    )

    # Plot the predicted data again in smaller bins, to show the distribution
    n_bins = 120
    ax.hist(
        np.tile(y_pred, int(n_bins / len(bin_edges))),
        bins=n_bins,
        alpha=0.2,
        label="Predicted Value Distribution",
    )

    # Set axes
    if y_name == "SPEED":
        ax.set_ylim(0, 1.25e6)
        ax.set_xlabel("Speed (m/s)")
    elif y_name == "INCLINE":
        ax.set_xlim(-20, 20)
        ax.set_ylim(0, 2.2e6)
        ax.set_xlabel("Incline (°)")

    ax.set_ylabel("Count")
    # plt.legend()
    plt.subplots_adjust(bottom=0.2, left=0.2)
    return fig

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer

app = typer.Typer()


@app.command()
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
    # plt.rcParams.update({"font.size": 16})
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
        # ax.set_xlim(0.5, 3.5)
        # ax.set_ylim(0, 1.4e6)
        ax.set_xlabel("Speed (m/s)")
    elif y_name == "INCLINE":
        # ax.set_xlim(-20, 20)
        # ax.set_ylim(0, 2.2e6)
        ax.set_xlabel("Incline (°)")

    ax.set_ylabel("Count")
    plt.legend()
    # plt.subplots_adjust(bottom=0.2, left=0.2)
    return fig


if __name__ == "__main__":
    app()

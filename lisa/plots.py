from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from loguru import logger
from tqdm import tqdm

from lisa.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def regression_histogram(df: pl.DataFrame, y_pred: np.ndarray, y_name: str) -> plt.Figure:
    """
    Plot a histogram of the predicted values versus the true values after regression.

    Args:
        df: The full DataFrame.
        y_pred: The predicted values.
        y_name: The name of the target column.

    Returns:
        fig: The matplotlib figure object.
    """
    # Normalise the histogram and bar scaling
    hist, bin_edges = np.histogram(y_pred, bins=50)
    bin_width = bin_edges[1] - bin_edges[0]
    # TODO replace .sort() with value_counts(sort=True)
    counts = df[y_name].value_counts().sort(y_name)
    bar_heights = counts["count"] / counts["count"].max() * hist.max()

    # Create the histogram and bar chart
    fig, ax = plt.subplots()
    y_name_label = y_name.capitalize()
    ax.hist(y_pred, bins=50, alpha=0.5, label=f"Predicted {y_name_label}")
    ax.bar(
        counts[y_name],
        bar_heights,
        color="red",
        alpha=0.5,
        label=f"Actual {y_name_label}",
        width=bin_width,
    )
    ax.set_xlabel(y_name_label)
    ax.set_ylabel("Frequency")
    ax.legend()

    return fig


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

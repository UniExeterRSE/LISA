import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from sklearn import metrics
from sklearn.base import BaseEstimator

from lisa.config import FOOT_SENSOR_PATTERN, IMU_PATTERN


def confusion_matrix(
    model: BaseEstimator,
    labels: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.DataFrame,
    savepath: Path = None,
) -> pl.DataFrame:
    """
    Generate a confusion matrix for the model and save it to a file if a savepath is provided.

    Args:
        model (BaseEstimator): The trained model.
        labels (pl.Series): The category labels.
        X_test (pl.DataFrame): The test features.
        y_test (pl.DataFrame): The test labels.
        savepath (Path, optional): The path to save the confusion matrix plot. Defaults to None.

    Returns:
        pl.Dataframe: The confusion matrix.

    """
    cm = metrics.confusion_matrix(y_test, model.predict(X_test), labels=labels, normalize="true")
    cm_df = pl.DataFrame(cm, schema=[str(label) for label in labels])
    cm_df = cm_df.with_columns(pl.Series("labels", labels))

    if savepath:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues_r", values_format=".2%", colorbar=False)
        all_sample_title = f"Score: {str(model.score(X_test, y_test))}"
        ax.set_title(all_sample_title, size=15)
        plt.savefig(savepath)
        plt.close(fig)

    return cm_df


def evaluate(model, labels, X_test, y_test):
    accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    score = model.score(X_test, y_test)
    cm = confusion_matrix(model, labels, X_test, y_test, False)

    return score, accuracy, cm


def analyse_feature_importances(json_path: Path) -> None:
    """
    Analyse the feature importances from a JSON file and print the aggregated scores for each component.
    Expects keys to be in the format '{statistic}_{measure}_{location}.{dimension}', i.e. 'min_gyro_Thigh_R.z'.

    Args:
        json_path (Path): Path to the JSON file containing the feature importances.

    Returns:
        None
    """
    # Load the JSON data
    with open(json_path) as f:
        feature_importances = json.load(f)

    # Initialize dictionaries to hold the aggregated scores for each component
    statistic_scores = defaultdict(float)
    measure_scores = defaultdict(float)
    location_scores = defaultdict(float)
    dimension_scores = defaultdict(float)

    # Regex to match the statistic, measure, location, and dimension parts of the key
    imu_pattern = re.compile(IMU_PATTERN)

    # Regex for foot sensor only, i.e. min_left foot sensor.lfs
    foot_sensor_pattern = re.compile(FOOT_SENSOR_PATTERN)

    # Parse the keys and aggregate the scores
    for key, importance in feature_importances.items():
        imu_match = imu_pattern.match(key)
        foot_sensor_match = foot_sensor_pattern.match(key)
        if imu_match:
            statistic, measure, location, dimension = imu_match.groups()
            statistic_scores[statistic] += importance
            measure_scores[measure] += importance
            location_scores[location] += importance
            dimension_scores[dimension] += importance
        elif foot_sensor_match:
            statistic, location = foot_sensor_match.groups()
            statistic_scores[statistic] += importance
            location_scores[location] += importance

    # Function to print sorted scores
    def print_sorted_scores(title, scores):
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"\nAggregated scores for each {title} (in order of importance):")
        for component, total_importance in sorted_scores:
            print(f"{component}: {total_importance:.4f}")

    # Print the aggregated scores for each component
    print_sorted_scores("statistic", statistic_scores)
    print_sorted_scores("measure", measure_scores)
    print_sorted_scores("location", location_scores)
    print_sorted_scores("dimension", dimension_scores)

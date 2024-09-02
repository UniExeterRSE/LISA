import matplotlib as plt
import seaborn as sns
from sklearn import metrics


def confusion_matrix(model, labels, X_test, y_test, save=False):
    cm = metrics.confusion_matrix(y_test, model.predict(X_test), labels=labels, normalize="true")
    # TODO use confusionmatrixdisplay?
    if save:
        plt.figure(figsize=(9, 9))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            linewidths=0.5,
            square=True,
            cmap="Blues_r",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")
        all_sample_title = f"Score: {str(model.score(X_test, y_test))}"
        plt.title(all_sample_title, size=15)
        # save fig

    return cm


def evaluate(model, labels, X_test, y_test):
    accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    score = model.score(X_test, y_test)
    cm = confusion_matrix(model, labels, X_test, y_test, False)

    return score, accuracy, cm

import matplotlib.pyplot as plt
from sklearn import metrics


def confusion_matrix(model, labels, X_test, y_test, savepath=None):
    cm = metrics.confusion_matrix(y_test, model.predict(X_test), labels=labels, normalize="true")
    if savepath:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues_r", values_format=".2%", colorbar=False)
        all_sample_title = f"Score: {str(model.score(X_test, y_test))}"
        ax.set_title(all_sample_title, size=15)
        plt.savefig(savepath)
        plt.close(fig)

    return cm


def evaluate(model, labels, X_test, y_test):
    accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
    score = model.score(X_test, y_test)
    cm = confusion_matrix(model, labels, X_test, y_test, False)

    return score, accuracy, cm

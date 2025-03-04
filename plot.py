import os

# from enum import Enum
# import numpy as np
import pickle

# import joblib
# import csv
import matplotlib.pyplot as plt

# import matplotlib.patches as mpatches
# import seaborn as sns
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    # confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    # accuracy_score,
    # balanced_accuracy_score,
    # f1_score,
    # recall_score,
    # precision_score,
)


def precision_recall_curve():
    # Compute precision-recall curves for each class separately
    precision_pre, recall_pre, _ = precision_recall_curve(
        Results.y_test, 1 - Results.y_scores, pos_label=0
    )
    precision_post, recall_post, _ = precision_recall_curve(
        Results.y_test, Results.y_scores, pos_label=1
    )

    # Compute macro-average precision-recall
    ap_pre = average_precision_score(
        Results.y_test, 1 - Results.y_scores, pos_label=0)
    ap_post = average_precision_score(
        Results.y_test, Results.y_scores, pos_label=1)
    ap_all = (ap_pre + ap_post) / 2  # Macro-average AP

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_pre,
        precision_pre,
        marker=".",
        label=f"Pre (Non-sleep deprived) {ap_pre:.4f}",
        color="pink",
    )
    plt.plot(
        recall_post,
        precision_post,
        label=f"Post (Sleep Deprived) {ap_post:.4f}",
        marker=".",
        color="red",
    )
    plt.plot(
        [0, 1],
        [ap_all, ap_all],
        linestyle="--",
        label=f"All (Macro AP={ap_all:.4f})",
        color="blue",
        linewidth=1,
    )

    # Labels and Legend
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Pre, Post, and All Classes)")
    plt.grid()

    # Save and Show Plot
    plt.savefig(
        os.path.join(plot_dir, "precision_recall_curve.png"), bbox_inches="tight"
    )
    plt.close()


def training_validation_loss():
    # TRAINING AND VALIDATION LOSS
    # Extract training and validation scores from GridSearchCV
    train_scores = Results.cv_results["mean_train_score"]
    val_scores = Results.cv_results["mean_test_score"]

    # Convert accuracy scores to loss (1 - accuracy)
    train_loss = 1 - train_scores
    val_loss = 1 - val_scores

    # Plot Training vs. Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(len(train_loss)),
        train_loss,
        marker="o",
        label="train",
        color="blue",
        linewidth=1,
    )
    plt.plot(
        range(len(val_loss)),
        val_loss,
        marker="s",
        label="validation",
        color="orange",
        linewidth=1,
    )

    # Labels and Legend
    plt.xlabel("Hyperparameter Index")
    plt.ylabel("Loss (1 - Accuracy)")
    plt.title("Training Loss vs. Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "train_loss_curve.png"),
                bbox_inches="tight")
    plt.close()


def perfomance_metrics():
    # Extract metrics from GridSearchCV
    train_accuracy = Results.cv_results["mean_train_score"]
    val_accuracy = Results.cv_results["mean_test_score"]

    train_precision = Results.cv_results.get(
        "mean_train_precision", None
    )  # If precision is available
    val_precision = Results.cv_results.get("mean_test_precision", None)

    train_recall = Results.cv_results.get("mean_train_recall", None)
    val_recall = Results.cv_results.get("mean_test_recall", None)

    train_f1 = Results.cv_results.get("mean_train_f1", None)
    val_f1 = Results.cv_results.get("mean_test_f1", None)

    train_bal_acc = Results.cv_results.get(
        "mean_train_balanced_accuracy", None)
    val_bal_acc = Results.cv_results.get("mean_test_balanced_accuracy", None)

    # Define metrics and their labels
    metrics = {
        "accuracy": (train_accuracy, val_accuracy),
        "precision": (train_precision, val_precision),
        "recall": (train_recall, val_recall),
        "f1-score": (train_f1, val_f1),
        "balanced_accuracy": (train_bal_acc, val_bal_acc),
    }

    # Plot all metrics
    for metric_name, (train_metric, val_metric) in metrics.items():
        if (
            train_metric is not None and val_metric is not None
        ):  # Only plot if data exists
            plt.figure(figsize=(8, 6))
            plt.plot(
                range(len(train_metric)),
                train_metric,
                marker="o",
                label=f"metrics/{metric_name}",
                color="blue",
                linewidth=1,
            )
            # plt.plot(
            #     range(len(val_metric)),
            #     val_metric,
            #     marker="s",
            #     label=f"val/",
            #     color="red",
            #     linewidth=1,
            # )

            # Labels and Legend
            # plt.xlabel("Hyperparameter Index")
            # plt.ylabel(metric_name)
            # plt.title(f"Training vs. Validation {metric_name}")
            # plt.legend()
            # plt.grid()

            # Save and Show Plot
            plot_filename = f"metrics_{
                metric_name.lower().replace(' ', '_')}.png"
            plt.savefig(os.path.join(plot_dir, plot_filename),
                        bbox_inches="tight")
            plt.close()


data_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/output/day_session/training_results/"
)

plot_dir = "../output/day_session/plots/"

with open(os.path.join(data_path, "results.pkl"), "rb") as f:
    data = pickle.load(f)


class Results:
    # Ys
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_pred = data["y_pred"]
    y_scores = data["y_scores"]

    # Xs
    x_train = data["X_train"]
    x_test = data["X_test"]
    x_train_pca = data["X_train_pca"]
    x_test_pca = data["X_train_pca"]

    # perfomance metrics
    f1_score = data["f1_score"]
    recall_score = data["recall_score"]
    precision_score = data["precision_score"]
    balanced_accuracy = data["balanced_accuracy"]
    specificity = data["specificity"]

    # confusion matrix
    conf_matrix = data["conf_matrix"]

    # cross validation
    cv_results = data["cv_results"]


print(Results.y_test)


def main():
    precision_recall_curve()
    training_validation_loss()
    perfomance_metrics()


if __name__ == "__main__":
    main()

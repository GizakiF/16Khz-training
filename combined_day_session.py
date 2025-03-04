import os
import numpy as np
import pickle
import joblib
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    precision_recall_curve,
)


def contain_all_results():
    data = {
        # Labels (true, predicted, scores)
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_scores": y_scores,
        # Features (raw & PCA-transformed)
        "X_train": X_train,
        "X_test": X_test,
        "X_train_pca": X_train_pca,
        "X_test_pca": X_test_pca,
        # Performance Metrics
        "f1_score": f1,
        "recall_score": recall,
        "precision_score": precision,
        "balanced_accuracy": balanced_acc,
        "specificity": specificity,
        "conf_matrix": confusion_matrix(y_test, y_pred),
        # Grid Search Resultsresults_,
        "cv_results": clf.cv_results_,
        "clf": clf,
    }

    # Save to pickle
    pickle_path = os.path.join(
        "../output/day_session/training_results", "results.pkl")
    # Ensure directory exists
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

    # Save data to pickle file
    with open(pickle_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ Results saved to {pickle_path}")


# Load Data
data_path = os.path.expanduser(
    "~/Research/Sleep Deprivation Detection using voice/dataset/osf/data.pkl"
)
with open(data_path, "rb") as f:
    data = pickle.load(f)

# Extract features and labels
tabStrf = np.asarray(data["tabStrf"])  # Features
tabSession = np.asarray(data["tabSession"])  # Labels ('pre', 'post')
tabDaySession = np.asarray(data["tabDaySession"])  # Day session numbers

# Convert Labels ('pre' → 0, 'post' → 1)
label_encoder = LabelEncoder()
tabSession_encoded = label_encoder.fit_transform(tabSession)

# Define correct day groupings (ensures both 'pre' and 'post' exist in each set)
day_groups = [(1, 4), (2, 5), (3, 6)]

# Set number of iterations
num_iterations = 1


# Run 10 iterations
for iteration in range(1, num_iterations + 1):
    print(f"\n🔄 Starting Iteration {iteration} / {num_iterations}")

    # Set iteration-specific directories
    if num_iterations > 1:
        iteration_dir = f"../output/iteration_{iteration}"
    else:
        iteration_dir = f"../output/day_session/"

    model_dir = os.path.join(iteration_dir, "models")
    plot_dir = os.path.join(iteration_dir, "plots")
    csv_dir = os.path.join(iteration_dir, "csv_results")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # CSV file for storing performance metrics (separate file per iteration)
    csv_file_path = os.path.join(
        csv_dir, f"model_performance_iter{iteration}.csv")

    # Prepare CSV file (overwrite for each iteration)
    with open(csv_file_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Iteration",
                "Day Group",
                "F1-Score",
                "Specificity",
                "Recall",
                "Precision",
                "Balanced Accuracy",
            ]
        )

    # Train and save model for each day group

    # Select data for the merged days
    # mask = np.isin(tabDaySession, group)

    # all day session combined to one


# Select all data from all day sessions
# Flatten groups
mask = np.isin(tabDaySession, [day for group in day_groups for day in group])
X = tabStrf[mask]
y = tabSession_encoded[mask]

# Check if both classes are present
unique_classes, class_counts = np.unique(y, return_counts=True)
print(
    f"🔍 Combined Data - Unique Classes: {
        dict(zip(label_encoder.inverse_transform(unique_classes), class_counts))
    }"
)

if len(unique_classes) < 2:
    print(f"⚠️ Skipping training (only one class present)")
else:
    # Train-test split (shuffle different for each iteration)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=iteration * 42, stratify=y
    )

    # PCA for dimensionality reduction
    n_components = 250
    pca = PCA(n_components=n_components, whiten=True)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Save PCA model
    pca_filename = os.path.join(model_dir, f"pca_combined.pkl")
    joblib.dump(pca, pca_filename)

    # Train SVM with GridSearchCV
    param_grid = {"C": np.logspace(-3, 3, num=5),
                  "gamma": np.logspace(-3, 3, num=5)}
    clf = GridSearchCV(
        SVC(kernel="rbf", probability=True),
        param_grid,
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=10),
        n_jobs=-1,
        return_train_score=True,
    )

    clf.fit(X_train_pca, y_train)

    # Save SVM model
    svm_filename = os.path.join(model_dir, f"svm_combined.pkl")
    joblib.dump(clf.best_estimator_, svm_filename)

    # Predictions and evaluation
    y_pred = clf.predict(X_test_pca)
    y_scores = clf.predict_proba(X_test_pca)[:, 1]  # Get probability scores

    # Compute performance metrics
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Compute specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Save metrics to CSV
    with open(csv_file_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Iteration",
                "Dataset",
                "F1-Score",
                "Specificity",
                "Recall",
                "Precision",
                "Balanced Accuracy",
            ]
        )
        writer.writerow(
            [iteration, "Combined", f1, specificity,
                recall, precision, balanced_acc]
        )

    # Save Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Combined Data (Iteration {iteration})")
    plt.savefig(
        os.path.join(plot_dir, f"confusion_matrix_combined.png"), bbox_inches="tight"
    )
    plt.close()
    contain_all_results()


print(f"✅ Iteration {iteration} - Model for Combined Data saved successfully!")
print(
    f"\n🎉 All models, plots, and metrics saved per iteration! Saved to: 📂 {momodel_dir}")
)

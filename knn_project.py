# ============================================
# KNN HEART DISEASE PREDICTION PROJECT
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from pathlib import Path

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ============================================
# 1. LOAD DATASET
# ============================================

print("\nLoading Heart Disease Dataset...")

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()

print("\nDataset Information")
print("-------------------")

print("Dataset Shape:", X.shape)
print("Number of Features:", X.shape[1])


# ============================================
# 2. EXPLORE DATASET
# ============================================

print("\nFirst 5 Rows:")
print(X.head())

print("\nDataset Information:")
X.info()

print("\nTarget Distribution Before Processing:")
print(y["num"].value_counts())


# ============================================
# 3. PREPROCESSING
# ============================================

# Convert target to binary classification
# 0 = No Heart Disease
# 1 = Heart Disease

y = (y["num"] > 0).astype(int)

print("\nTarget Distribution After Processing:")
print(y.value_counts())

# Missing values

print("\nMissing Values:")
print(X.isnull().sum())

# Duplicate rows

print("\nDuplicate Rows:", X.duplicated().sum())


# ============================================
# 4. SPLIT DATASET
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================
# 5. CREATE KNN PIPELINE
# ============================================

def create_knn_model(k):

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])


# ============================================
# 6. TRY DIFFERENT K VALUES
# ============================================

print("\nTesting Different K Values")
print("---------------------------")

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

accuracies = []

for k in k_values:

    model = create_knn_model(k)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)

    accuracies.append(score)

    print(
        f"K = {k}, "
        f"Accuracy = {score * 100:.2f}%"
    )


# ============================================
# 7. CROSS VALIDATION
# ============================================

print("\nCross Validation Results")
print("------------------------")

cv_results = []

for k in k_values:

    model = create_knn_model(k)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    mean_score = scores.mean()

    cv_results.append(mean_score)

    print(
        f"K = {k}, "
        f"Cross Validation Accuracy = {mean_score * 100:.2f}%"
    )


# ============================================
# 8. SELECT BEST K
# ============================================

best_index = cv_results.index(max(cv_results))

best_k = k_values[best_index]

best_cv_accuracy = cv_results[best_index]

print("\nBest K Value:", best_k)

print(
    "Best Cross Validation Accuracy:",
    round(best_cv_accuracy * 100, 2),
    "%"
)


# ============================================
# 9. TRAIN FINAL MODEL
# ============================================

print("\nTraining Final KNN Model...")

final_model = create_knn_model(best_k)

final_model.fit(X_train, y_train)

final_predictions = final_model.predict(X_test)

final_accuracy = accuracy_score(
    y_test,
    final_predictions
)

print("\nFinal Model Results")
print("-------------------")

print("Selected K:", best_k)

print(
    "Test Accuracy:",
    round(final_accuracy * 100, 2),
    "%"
)


# ============================================
# 10. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    final_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================
# 11. CLASSIFICATION REPORT
# ============================================

report = classification_report(
    y_test,
    final_predictions,
    target_names=[
        "No Heart Disease",
        "Heart Disease"
    ],
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "No Heart Disease",
            "Heart Disease"
        ],
        zero_division=0
    )
)


# ============================================
# 12. CREATE PLOTS FOLDER
# ============================================

Path("plots").mkdir(exist_ok=True)


# ============================================
# 13. K VALUES COMPARISON PLOT
# ============================================

plt.figure(figsize=(9, 5))

plt.plot(
    k_values,
    [x * 100 for x in accuracies],
    marker="o",
    linewidth=2,
    label="Test Accuracy"
)

plt.title("KNN Accuracy for Different K Values")

plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")

plt.xticks(k_values)

plt.grid(alpha=0.3)

plt.legend()

plt.savefig(
    "plots/knn_k_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 14. CROSS VALIDATION PLOT
# ============================================

plt.figure(figsize=(9, 5))

plt.plot(
    k_values,
    [x * 100 for x in cv_results],
    marker="o",
    linewidth=2,
    label="Cross Validation Accuracy"
)

plt.title("KNN Cross Validation Performance")

plt.xlabel("K Value")
plt.ylabel("Mean Accuracy (%)")

plt.xticks(k_values)

plt.grid(alpha=0.3)

plt.legend()

plt.savefig(
    "plots/knn_cross_validation.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 15. CONFUSION MATRIX PLOT
# ============================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Disease",
        "Disease"
    ],
    yticklabels=[
        "No Disease",
        "Disease"
    ]
)

plt.title("Final KNN Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "plots/knn_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 16. SAVE FINAL MODEL
# ============================================

joblib.dump(
    final_model,
    "knn_model.pkl"
)

print("\nFinal KNN Model Saved Successfully!")


# ============================================
# 17. SAVE MODEL METRICS
# ============================================

metrics = {

    "accuracy": float(final_accuracy),

    "best_k": int(best_k),

    "cv_accuracy": float(best_cv_accuracy),

    "classification_report": report,

    "confusion_matrix": cm.tolist(),

    "k_values": k_values,

    "test_accuracies": [
        float(x) for x in accuracies
    ],

    "cv_accuracies": [
        float(x) for x in cv_results
    ]
}


with open("metrics.json", "w") as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================
# 18. PROJECT COMPLETED
# ============================================

print("\n================================")

print("KNN PROJECT COMPLETED SUCCESSFULLY!")

print("================================")

print("Best K:", best_k)

print(
    "Final Accuracy:",
    round(final_accuracy * 100, 2),
    "%"
)

print("\nGenerated Files:")

print("1. knn_model.pkl")
print("2. metrics.json")
print("3. plots/knn_k_comparison.png")
print("4. plots/knn_cross_validation.png")
print("5. plots/knn_confusion_matrix.png")

print("\nAll Files Are Ready For Streamlit UI!")
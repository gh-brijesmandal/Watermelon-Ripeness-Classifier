"""
evaluate_models.py

Model evaluation for the Watermelon Ripeness Classifier project.
Built directly against Watermelon_Features.xlsx's actual columns.

KEY DESIGN CHOICE: splits are grouped by melon_id, not by row.
Each melon produces ~10 correlated taps. If taps from the same melon
land in both train and test, the model can partly "recognize the melon"
instead of learning ripeness in general, and your accuracy will look
better than it really is. GroupKFold / GroupShuffleSplit prevent this
by keeping every tap from one melon on the same side of the split.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# 1. LOAD + CLEAN
# -------------------------------------------------------------
df = pd.read_excel("Watermelon_Features.xlsx")

df["LABEL"] = df["LABEL"].str.strip()  # fixes "ripe " -> "ripe"

feature_cols = [
    "peak_amplitude", "rms_energy", "zero_crossing_rate",
    "damping_coefficient", "peak_frequency_hz",
    "spectral_skewness", "spectral_kurtosis", "f_max_hz",
]

X = df[feature_cols]
y = df["LABEL"]
groups = df["melon_id"]  # the grouping key that keeps melons intact across splits

print("Class counts (taps):")
print(y.value_counts())
print("\nMelons per class:")
print(df.groupby("LABEL")["melon_id"].nunique())

# -------------------------------------------------------------
# 2. GROUPED TRAIN/TEST SPLIT
# -------------------------------------------------------------
# With only 9 melons total, don't hold out too many - GroupShuffleSplit
# with 1 split, ~25% of melons in test.
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 3. GROUPED CROSS-VALIDATION
# -------------------------------------------------------------
# With ~7 melons left in the training set, keep folds small.
n_groups_train = groups_train.nunique()
n_splits = min(4, n_groups_train)  # never ask for more folds than melons available
gkf = GroupKFold(n_splits=n_splits)

svm_cv_scores = cross_val_score(
    SVC(kernel="rbf"), X_train_scaled, y_train, groups=groups_train, cv=gkf
)
rf_cv_scores = cross_val_score(
    RandomForestClassifier(random_state=42), X_train, y_train, groups=groups_train, cv=gkf
)

print(f"\n=== Grouped Cross-Validation Accuracy ({n_splits}-fold, by melon) ===")
print(f"SVM: {svm_cv_scores.mean():.3f} +/- {svm_cv_scores.std():.3f}  (scores: {svm_cv_scores.round(3)})")
print(f"Random Forest: {rf_cv_scores.mean():.3f} +/- {rf_cv_scores.std():.3f}  (scores: {rf_cv_scores.round(3)})")
print("\nNote: with only 9 melons total, these numbers will move around a lot.")
print("Treat them as a sanity check on the pipeline, not a final result yet -")
print("re-run this once your remaining 5 melons are added.")

# -------------------------------------------------------------
# 4. HYPERPARAMETER TUNING (grouped CV inside the search too)
# -------------------------------------------------------------
svm_param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"],
}
rf_param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=gkf, scoring="accuracy")
svm_grid.fit(X_train_scaled, y_train, groups=groups_train)

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=gkf, scoring="accuracy")
rf_grid.fit(X_train, y_train, groups=groups_train)

print("\n=== Best Hyperparameters ===")
print("SVM:", svm_grid.best_params_)
print("Random Forest:", rf_grid.best_params_)

# -------------------------------------------------------------
# 5. FINAL EVALUATION ON HELD-OUT MELONS (unseen in training)
# -------------------------------------------------------------
best_svm = svm_grid.best_estimator_
best_rf = rf_grid.best_estimator_

svm_preds = best_svm.predict(X_test_scaled)
rf_preds = best_rf.predict(X_test)

for name, preds in [("SVM", svm_preds), ("Random Forest", rf_preds)]:
    print(f"\n=== {name} - Held-out melon performance ===")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print("Confusion Matrix (labels order: alphabetical):")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

# -------------------------------------------------------------
# 6. FEATURE IMPORTANCE (Random Forest)
# -------------------------------------------------------------
importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Random Forest Feature Importances ===")
print(importances)
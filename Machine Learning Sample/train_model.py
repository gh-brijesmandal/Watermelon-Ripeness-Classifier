"""
train_model.py
Trains a machine learning classifier to predict watermelon ripeness
(unripe / ripe / overripe) from the acoustic features main.py already
logs to Watermelon_Features.xlsx.

USAGE
-----
    python train_model.py
        Trains on every labeled row currently in the Excel file,
        prints a cross-validated performance report, and saves the
        trained model to ripeness_model.joblib.

    python train_model.py --predict-unlabeled
        Also runs the freshly trained model on every row that does NOT
        have a usable label yet, and prints what it currently thinks
        each one is (Ripe / Unripe / Overripe). Handy for a sanity
        check while you're still labeling data.

    python train_model.py --min-per-class N
        Change the minimum samples-per-class needed before the script
        will trust a stratified cross-validation split (default 2).

Dependencies (install once):
    pip install scikit-learn joblib pandas openpyxl
"""

import argparse
import re
import sys

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


EXCEL_PATH = "Watermelon_Features.xlsx"
MODEL_PATH = "ripeness_model.joblib"
LABEL_COLUMN = "LABEL"


FEATURE_COLUMNS = [
    "peak_amplitude",
    "rms_energy",
    "zero_crossing_rate",
    "damping_coefficient",
    "peak_frequency_hz",
    "spectral_skewness",
    "spectral_kurtosis",
    "f_max_hz",
]

CANONICAL_CLASSES = ["unripe", "ripe", "overripe"]


 
# Label cleanup

def normalize_label(raw_label):
    """
    Maps whatever messy text is in the LABEL column to one of
    'unripe' / 'ripe' / 'overripe', or None if it can't be confidently
    mapped (blank, NaN, or something ambiguous like "needs verification").

    Checks 'overripe' and 'unripe' before bare 'ripe', since both of
    those strings contain "ripe" as a substring.
    """
    if pd.isna(raw_label):
        return None

    text = str(raw_label).strip().lower()
    if text == "":
        return None

    if "overripe" in text or "over ripe" in text or "over-ripe" in text:
        return "overripe"
    if "unripe" in text or "un ripe" in text or "under ripe" in text or "under-ripe" in text:
        return "unripe"
    if re.search(r"\bripe\b", text):
        return "ripe"

    # Things like "needs verification" or "slightly unripe" (caught above)
    # fall through here only if truly ambiguous.
    return None


 
# Data loading
 
def load_dataset(excel_path=EXCEL_PATH):
    """
    Reads the Excel log and splits it into:
      - labeled_df: rows with a usable ripeness label (used for training)
      - unlabeled_df: rows with no/ambiguous label (candidates to predict)
    Both are returned alongside the cleaned 'ripeness' column added to labeled_df.
    """
    df = pd.read_excel(excel_path)

    # Drop stray empty "Unnamed: N" columns Excel sometimes leaves behind.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Excel file is missing expected feature columns: {missing_features}. "
            f"Did the column names in main.py change?"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"No '{LABEL_COLUMN}' column found in {excel_path}. "
            f"Add one and fill in ripeness labels (unripe/ripe/overripe) per row."
        )

    df["ripeness"] = df[LABEL_COLUMN].apply(normalize_label)

    # A row also needs all its features present to be usable.
    has_features = df[FEATURE_COLUMNS].notna().all(axis=1)

    labeled_df = df[df["ripeness"].notna() & has_features].copy()
    unlabeled_df = df[df["ripeness"].isna() | ~has_features].copy()

    skipped_ambiguous = df[LABEL_COLUMN].notna() & df["ripeness"].isna()
    if skipped_ambiguous.any():
        print(
            f"Note: {skipped_ambiguous.sum()} row(s) had a LABEL value that "
            f"couldn't be confidently mapped to unripe/ripe/overripe and were "
            f"skipped for training, e.g.: "
            f"{df.loc[skipped_ambiguous, LABEL_COLUMN].unique().tolist()}"
        )

    return labeled_df, unlabeled_df


 
# Model training
 
def build_pipeline(model):
    """Impute missing feature values, standardize, then classify."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", model),
    ])


def choose_cv(y, min_per_class, encoder=None):
    """
    Picks a cross-validation strategy that actually fits the amount of
    data available. With very few labeled taps (which is expected early
    on), a 5-fold stratified split isn't possible, so this scales down
    automatically, down to leave-one-out if needed.
    """
    class_counts = pd.Series(y).value_counts()
    smallest_class = class_counts.min()

    if smallest_class < min_per_class:
        smallest_label = class_counts.idxmin()
        if encoder is not None:
            smallest_label = encoder.inverse_transform([smallest_label])[0]
        raise ValueError(
            f"Class '{smallest_label}' only has {smallest_class} labeled "
            f"sample(s), below the minimum of {min_per_class}. Label a few more "
            f"taps for that class before training (or lower --min-per-class)."
        )

    n_splits = min(5, smallest_class)
    if n_splits < 2:
        print(
            "Very few labeled samples per class -- falling back to "
            "leave-one-out cross-validation. Treat these results as a rough "
            "sanity check, not a real accuracy estimate, until you have more data."
        )
        return LeaveOneOut()

    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def train_and_evaluate(labeled_df, min_per_class=2):
    X = labeled_df[FEATURE_COLUMNS].values
    y_raw = labeled_df["ripeness"].values

    encoder = LabelEncoder()
    encoder.fit(CANONICAL_CLASSES)  # fixed class order, even if one is unseen so far
    y = encoder.transform(y_raw)

    present_classes = sorted(set(y_raw))
    print(f"Training on {len(labeled_df)} labeled taps across classes: {present_classes}")
    if len(present_classes) < 2:
        raise ValueError(
            "Only one ripeness class is labeled so far -- need at least two "
            "classes (e.g. some 'ripe' and some 'unripe' taps) to train a classifier."
        )

    cv = choose_cv(y, min_per_class, encoder=encoder)

    candidates = {
        "svm_rbf": build_pipeline(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        "random_forest": build_pipeline(
            RandomForestClassifier(n_estimators=200, random_state=42)
        ),
    }

    best_name, best_pipeline, best_acc = None, None, -1.0
    for name, pipeline in candidates.items():
        preds = cross_val_predict(pipeline, X, y, cv=cv)
        acc = accuracy_score(y, preds)
        print(f"\n--- {name} (cross-validated) ---")
        print(f"Accuracy: {acc:.3f}")
        print(classification_report(
            y, preds,
            labels=encoder.transform([c for c in CANONICAL_CLASSES if c in present_classes]),
            target_names=[c for c in CANONICAL_CLASSES if c in present_classes],
            zero_division=0,
        ))
        if acc > best_acc:
            best_name, best_pipeline, best_acc = name, pipeline, acc

    print(f"Best model: {best_name} (cross-validated accuracy {best_acc:.3f})")

    # Refit the winning model on ALL labeled data for the version we actually save.
    best_pipeline.fit(X, y)

    bundle = {
        "pipeline": best_pipeline,
        "label_encoder": encoder,
        "feature_columns": FEATURE_COLUMNS,
        "model_name": best_name,
        "cv_accuracy": best_acc,
    }
    dump(bundle, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    return bundle


 
# Inference
 
def predict_ripeness(bundle, feature_row):
    """
    feature_row: dict or pandas Series with at least the keys in
    FEATURE_COLUMNS. Returns the predicted label string
    ('unripe' / 'ripe' / 'overripe').
    """
    x = np.array([[feature_row[col] for col in bundle["feature_columns"]]], dtype=float)
    pred_encoded = bundle["pipeline"].predict(x)[0]
    return bundle["label_encoder"].inverse_transform([pred_encoded])[0]


def predict_unlabeled_rows(bundle, unlabeled_df):
    usable = unlabeled_df[unlabeled_df[FEATURE_COLUMNS].notna().all(axis=1)].copy()
    if usable.empty:
        print("No unlabeled rows with complete features to predict on.")
        return

    print(f"\nPredictions for {len(usable)} not-yet-labeled tap(s):")
    for _, row in usable.iterrows():
        label = predict_ripeness(bundle, row)
        melon = row.get("melon_id", "?")
        tap = row.get("tap_id", "?")
        source = row.get("source_file", "?")
        print(f"  melon_id={melon} tap_id={tap} ({source}) -> predicted: {label.upper()}")


 
# Main
 
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict-unlabeled", action="store_true",
        help="After training, predict ripeness for rows that don't have a usable LABEL yet.",
    )
    parser.add_argument(
        "--min-per-class", type=int, default=2,
        help="Minimum labeled samples required per class before training (default: 2).",
    )
    parser.add_argument(
        "--excel-path", default=EXCEL_PATH,
        help=f"Path to the features Excel file (default: {EXCEL_PATH}).",
    )
    args = parser.parse_args()

    labeled_df, unlabeled_df = load_dataset(args.excel_path)

    if labeled_df.empty:
        sys.exit(
            f"No usable labeled rows found. Fill in the '{LABEL_COLUMN}' column "
            f"in {args.excel_path} with unripe/ripe/overripe for at least a few "
            f"taps per class, then re-run this script."
        )

    bundle = train_and_evaluate(labeled_df, min_per_class=args.min_per_class)

    if args.predict_unlabeled:
        predict_unlabeled_rows(bundle, unlabeled_df)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Classify organoids from their morphology features (the downstream task).

A deliberately tiny scikit-learn model on top of morphology.csv: train a
RandomForest on the shape features to predict each organoid's survey label. The
point in the workshop is the pipeline shape, not the score.

The labels CSV has columns `image_id,label[,organoid_id]`. If `organoid_id` is
present, cross-validation groups by organoid (GroupKFold) so the same organoid
never lands in both train and test -- essential when labels were propagated
across an organoid's days, or accuracy is meaninglessly inflated by leakage.

    python src/classify.py --features runs/morphology.csv --labels labels.csv \
        --out runs/classified.csv
"""
import argparse
import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold

FEAT_COLS = ["area_px", "perimeter_px", "circularity", "eccentricity"]


def parse_command_line_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True, help="CSV: image_id,label[,organoid_id]")
    ap.add_argument("--out", default="runs/classified.csv")
    args = ap.parse_args()
    return args


def classify_supervised(morph_df: pd.DataFrame, labels_path: str) -> pd.DataFrame:
    """Train a RandomForest on morphology features to predict the survey label.

    Joins features to labels on `image_id` and reports cross-validated accuracy.
    Uses GroupKFold by `organoid_id` when that column is present (leakage-safe),
    otherwise StratifiedKFold. Fold count is clamped down for small datasets.

    Args:
        morph_df: Morphology features, one row per image (must have `image_id`).
        labels_path: CSV with `image_id,label[,organoid_id]`.

    Returns:
        The merged frame with an added `prediction` column.
    """
    labels = pd.read_csv(labels_path)  # Load in labels
    merged_df = morph_df.merge(labels, on="image_id")  # Merge labels with morphology data
    if merged_df.empty:
        raise SystemExit("[classify] no rows after joining features to labels "
                         "(image_id mismatch?) -- nothing to train on")

    X, y = merged_df[FEAT_COLS], merged_df["label"]  # Pull out the feature matrix and target labels
    clf = RandomForestClassifier(n_estimators=200, random_state=0)  # Define a 200-tree RandomForest

    if "organoid_id" in merged_df.columns:
        n_splits = max(2, min(5, merged_df["organoid_id"].nunique()))
        cv = GroupKFold(n_splits=n_splits)  # Ensure all rows of the same organoid are in the same fold
        merged_df["prediction"] = cross_val_predict(clf, X, y, cv=cv, groups=merged_df["organoid_id"])
        scheme = f"GroupKFold({n_splits}) by organoid"
    else:
        n_splits = max(2, min(5, int(y.value_counts().min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        merged_df["prediction"] = cross_val_predict(clf, X, y, cv=cv)
        scheme = f"StratifiedKFold({n_splits}) [no organoid_id -> may leak]"

    accuracy = (merged_df["prediction"] == y).mean()
    print(f"[classify] supervised RF, {scheme}: accuracy={accuracy:.3f} on {len(merged_df)} labeled")
    return merged_df


def main() -> None:
    """Train a RandomForest on morphology features to predict the survey label."""
    start = datetime.datetime.now()

    # Command line arguments
    args = parse_command_line_args()

    morph_df = pd.read_csv(args.features)
    result = classify_supervised(morph_df, args.labels)

    result.to_csv(args.out, index=False)
    print(f"[classify] wrote {args.out}")

    end = datetime.datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()

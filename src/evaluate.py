#!/usr/bin/env python3
"""Evaluate and plot the classified organoid results.

Consumes classify.py's output (image_id, features, label, prediction) and
produces the evaluation the Promega paper emphasizes for this imbalanced QC
task: a confusion matrix, per-class precision/recall/F1, balanced accuracy, and
the true-negative rate on the minority "Not Acceptable" class -- plus a
RandomForest feature-importance plot (which shape feature drives the call).

If the input carries a `dayID` column (threaded through by make_labels.py), it
also breaks balanced accuracy down by culture day -- showing how quality becomes
more predictable toward Dy30, like the production shape-metrics notebook.

Mirrors the conventions in the production analysis (sklearn ConfusionMatrixDisplay
with a Blues colormap; feature-importance bar chart).

    python src/evaluate.py --classified runs/classified.csv --out-dir runs/eval
"""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs on a cluster node, no display
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

FEAT_COLS = ["area_px", "perimeter_px", "circularity", "eccentricity"]
CLASSES = ["Not Acceptable", "Acceptable"]   # negative (minority) first


def _day_number(day_id: str) -> int:
    """Extract the integer day from a dayID like 'Dy30' -> 30 (for ordering)."""
    digits = "".join(ch for ch in str(day_id) if ch.isdigit())
    return int(digits) if digits else 0


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute the headline metrics for the imbalanced QC task.

    Args:
        y_true: Ground-truth survey labels.
        y_pred: Predicted labels (out-of-sample from classify.py).

    Returns:
        Dict with accuracy, balanced accuracy, true-negative rate on
        "Not Acceptable", and the full per-class report.
    """
    return {
        "n": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # TNR = recall on the minority "Not Acceptable" class (the QC priority)
        "tnr_not_acceptable": float(recall_score(y_true, y_pred, pos_label="Not Acceptable")),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def plot_confusion(y_true: pd.Series, y_pred: pd.Series, out: Path) -> None:
    """Save a Blues confusion-matrix PNG (Promega-style).

    Args:
        y_true: Ground-truth survey labels.
        y_pred: Predicted labels from classify.py.
        out: Destination .png path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Organoid quality — confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, out: Path) -> None:
    """Refit a RandomForest on all rows and bar-plot its feature importances.

    The classifier in classify.py is cross-validated (no single saved model), so
    we refit on the full labelled set purely to read feature_importances_.

    Args:
        df: Classified frame with the FEAT_COLS feature columns and `label`.
        out: Destination .png path.
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(df[FEAT_COLS], df["label"])
    order = clf.feature_importances_.argsort()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEAT_COLS[i] for i in order], clf.feature_importances_[order], color="steelblue")
    ax.set_xlabel("importance")
    ax.set_title("RandomForest feature importance")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accuracy_by_day(df: pd.DataFrame) -> dict[str, dict]:
    """Balanced accuracy and count per culture day, ordered Dy03 -> Dy30.

    Args:
        df: Classified frame with `dayID`, `label`, `prediction` columns.

    Returns:
        Ordered dict {dayID: {"balanced_accuracy": float, "n": int,
        "single_class": bool}}; `single_class` flags days where only one label
        is present, so their balanced accuracy is degenerate.
    """
    out = {}
    for day in sorted(df["dayID"].unique(), key=_day_number):
        sub = df[df["dayID"] == day]
        with warnings.catch_warnings():  # quiet single-class small-day warning
            warnings.simplefilter("ignore")
            bal = float(balanced_accuracy_score(sub["label"], sub["prediction"]))
        out[str(day)] = {
            "balanced_accuracy": bal,
            "n": int(len(sub)),
            "single_class": bool(sub["label"].nunique() < 2),  # bal is degenerate if True
        }
    return out


def plot_accuracy_by_day(per_day: dict[str, dict], out: Path) -> None:
    """Bar-plot balanced accuracy per culture day (predictability over time).

    Args:
        per_day: Output of accuracy_by_day (dayID -> balanced_accuracy/n/...).
        out: Destination .png path.
    """
    days = list(per_day)
    vals = [per_day[d]["balanced_accuracy"] for d in days]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(days, vals, color="steelblue")
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Quality predictability by culture day")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Evaluate classify.py output: write metrics JSON + confusion/importance PNGs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--classified", required=True, help="classify.py output CSV")
    ap.add_argument("--out-dir", default="runs/eval")
    args = ap.parse_args()

    df = pd.read_csv(args.classified)
    if "prediction" not in df.columns or "label" not in df.columns:
        raise SystemExit("[eval] input needs 'label' and 'prediction' columns "
                         "(run classify.py first)")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(df["label"], df["prediction"])
    plot_confusion(df["label"], df["prediction"], out_dir / "confusion_matrix.png")
    plot_feature_importance(df, out_dir / "feature_importance.png")
    plots = "confusion_matrix.png + feature_importance.png"

    # Per-day breakdown (only if labels carried dayID through).
    if "dayID" in df.columns:
        metrics["by_day"] = accuracy_by_day(df)
        plot_accuracy_by_day(metrics["by_day"], out_dir / "accuracy_by_day.png")
        plots += " + accuracy_by_day.png"

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[eval] n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
          f"balanced_acc={metrics['balanced_accuracy']:.3f}  "
          f"TNR(Not Acceptable)={metrics['tnr_not_acceptable']:.3f}")
    print(f"[eval] wrote metrics.json + {plots} -> {out_dir}")


if __name__ == "__main__":
    main()

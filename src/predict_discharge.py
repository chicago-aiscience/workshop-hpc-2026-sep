#!/usr/bin/env python3
"""Predict discharge for a index of reaches with the trained consensus model.

Loads the model saved by train_consensus.py and applies it to the inference
manifest, writing predicted discharge per (reach, overpass). Designed to run as a
SLURM job array (Lesson 4): each task takes a `--index` of the manifest rows so
the reaches are processed in parallel. Runs on GPU when one is available.

    python src/predict_discharge.py --manifest data/manifest_infer.csv \
        --model runs/consensus/consensus_model_final.pt \
        --out-dir runs/predictions --index 0 --num-tasks 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features import make_features           # noqa: E402
from train_consensus import ConsensusMLP      # noqa: E402
from utils import configure_logging

# Columns carried through to the prediction output so downstream stages can score
# the model against the gauge and against the naive-mean / consensus baselines.
# `q_consensus` is carried as a BASELINE only -- it is not a model input feature.
CARRY = ["reach_id", "basin", "date", "month",
         "q_metroman", "q_momma", "q_neobam", "q_sic4dvar", "q_consensus", "gauge_q"]

logger = configure_logging("predict_discharge")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a prediction index."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", type=int, default=0, help="this task's index (0-based)")
    ap.add_argument("--num-tasks", type=int, default=1, help="total number of array tasks")
    return ap.parse_args()


def load_model(path: str, feature_names: list[str], device: str) -> ConsensusMLP:
    """Rebuild the MLP and load its weights + normalization buffers.

    Asserts the checkpoint's feature order matches `feature_names` (what
    `make_features` produces now), so a model trained on a different feature set
    -- e.g. an old checkpoint that still included `q_consensus` -- fails loudly
    here instead of silently mispredicting on mismatched inputs.

    Args:
        path: The `consensus_model_final.pt` saved by training.
        feature_names: The feature column order from the current `make_features`.
        device: "cuda" or "cpu".

    Returns:
        The loaded model in eval mode.
    """
    state = torch.load(path, map_location=device)

    # Detect feature mismatch between training and prediction
    saved = state.get("feature_names")
    if saved is None:
        logger.warning("[predict] checkpoint has no feature_names; skipping feature check.")

    elif list(saved) != list(feature_names):
        raise SystemExit(
            f"[predict] feature mismatch: model was trained on {list(saved)} but "
            f"make_features now produces {list(feature_names)}. Retrain the model.")

    # Rebuild with the SAME hidden width training used (stamped in the checkpoint
    # config); fall back to the class default for older checkpoints.
    hidden = state.get("config", {}).get("model", {}).get("hidden", 64)

    # mean/std are restored from the checkpoint buffers, so seed with placeholders.
    n_features = len(feature_names)
    model = ConsensusMLP(n_features, np.zeros(n_features), np.ones(n_features),
                         hidden=hidden).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    return model


def main() -> None:
    """Predict discharge for this index's rows and write them to the output dir."""
    args = parse_args()
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.manifest)
    index = df.iloc[args.index::args.num_tasks].reset_index(drop=True)
    logger.info(f"[predict] device={device} index {args.index}/{args.num_tasks} rows={len(index)}")

    X, names = make_features(index)
    model = load_model(args.model, names, device)
    with torch.no_grad():
        log_q = model(torch.from_numpy(X).to(device)).cpu().numpy()
    index["predicted_q"] = np.expm1(log_q)               # invert the log1p target

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{args.index:03d}.csv"
    index[CARRY + ["predicted_q"]].to_csv(out_path, index=False)
    logger.info(f"[predict] wrote {out_path}")


if __name__ == "__main__":
    main()

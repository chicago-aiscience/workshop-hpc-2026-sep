#!/usr/bin/env python3
"""Plot the learned-consensus results: skill scores, scatter, and hydrographs.

Reads the aggregated discharge table and the scoreboard, and writes figures that
make the payoff visible -- does the learned consensus track the gauge better than
the naive multi-algorithm mean and Confluence's own consensus?

    python src/evaluate.py --discharge runs/discharge.csv \
        --scored runs/scored.csv --out-dir runs/eval
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

from utils import configure_logging

FLPE_COLS = ["q_metroman", "q_momma", "q_neobam", "q_sic4dvar"]
ESTIMATORS = ["learned", "naive_mean", "consensus"]

logger = configure_logging("benchmark_consensus")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation/plotting."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--discharge", required=True, help="aggregated discharge.csv")
    ap.add_argument("--scored", required=True, help="scored.csv from benchmark_consensus.py")
    ap.add_argument("--out-dir", required=True)
    return ap.parse_args()


def plot_skill(scored: pd.DataFrame, out: Path) -> None:
    """Bar chart of overall KGE and NSE for each estimator."""
    ov = scored[scored.scope == "overall"].set_index("estimator").reindex(ESTIMATORS)
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(ESTIMATORS))

    ax.bar(x - 0.2, ov["kge"], 0.4, label="KGE")
    ax.bar(x + 0.2, ov["nse"], 0.4, label="NSE")
    ax.set_xticks(x, ESTIMATORS)
    ax.axhline(0, color="k", lw=0.8)

    ax.set_ylabel("skill score (higher is better)")
    ax.set_title("Overall skill vs. gauge")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "skill_scores.png", dpi=130)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, out: Path) -> None:
    """Predicted-vs-observed scatter (log-log) for learned vs. naive mean."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    sims = {"learned": df["predicted_q"], "naive_mean": df[FLPE_COLS].mean(axis=1)}
    for ax, (name, sim) in zip(axes, sims.items()):
        o, s = df["gauge_q"].to_numpy(float), sim.to_numpy(float)
        m = np.isfinite(o) & np.isfinite(s) & (o > 0) & (s > 0)
        ax.scatter(o[m], s[m], s=10, alpha=0.4)
        lim = [max(1e-2, min(o[m].min(), s[m].min())), max(o[m].max(), s[m].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(xscale="log", yscale="log", xlabel="observed gauge Q (m³/s)", title=name)
    axes[0].set_ylabel("estimated Q (m³/s)")

    fig.suptitle("Estimated vs. observed discharge")
    fig.tight_layout()
    fig.savefig(out / "scatter_pred_vs_obs.png", dpi=130)

    plt.close(fig)


def plot_hydrograph(df: pd.DataFrame, out: Path) -> None:
    """Time series for the best-gauged reach: gauge vs. learned vs. consensus."""
    gauged = df[df["gauge_q"].notna()]
    if gauged.empty:
        return

    reach = gauged["reach_id"].value_counts().idxmax()
    r = df[(df.reach_id == reach) & df.gauge_q.notna()].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r.date, r.gauge_q, "k-o", ms=3, label="gauge (observed)")
    ax.plot(r.date, r.predicted_q, "-s", ms=3, label="learned consensus")
    ax.plot(r.date, r.q_consensus, "-^", ms=3, alpha=0.7, label="Confluence consensus")

    ax.set(ylabel="discharge (m³/s)", title=f"Reach {reach} hydrograph")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "hydrograph_best_reach.png", dpi=130)
    plt.close(fig)


def main() -> None:
    """Write the skill, scatter, and hydrograph figures plus a text summary."""
    args = parse_args()
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.discharge)
    scored = pd.read_csv(args.scored)

    plot_skill(scored, out)
    plot_scatter(df[df.gauge_q.notna()], out)
    plot_hydrograph(df, out)
    (out / "summary.txt").write_text(
        scored[scored.scope == "overall"].to_string(index=False) + "\n")
    logger.info(f"[evaluate] wrote figures + summary -> {out}")


if __name__ == "__main__":
    main()

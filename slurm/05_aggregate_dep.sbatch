#!/usr/bin/env bash
#SBATCH --job-name=consensus-aggregate
#SBATCH --partition=general          # CPU-only stage; no --gres needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/aggregate_%j.out
# --- Workshop 4: SLURM dependency. Run this ONLY after the prediction array
#     finishes, then submit the chain like:
#
#       aid=$(sbatch --parsable slurm/04_predict_array.sbatch)
#       sbatch --dependency=afterok:$aid slurm/05_aggregate_dep.sbatch
#
#     `afterok` = start only if every array task succeeded.

set -euo pipefail
source config/paths.sh
source activate "$ENV_NAME"

# Unbuffered stdout/stderr so aggregate/benchmark/evaluate output streams to the
# .out file live rather than being held in a buffer until the job ends.
export PYTHONUNBUFFERED=1

# Per-shard predictions (from the array) -> one tidy per-reach discharge table.
python src/aggregate_discharge.py \
    --pred-dir "$WORK_DIR/runs/predictions" \
    --out "$WORK_DIR/runs/discharge.csv"

# Score the learned consensus against the gauge TARGET, alongside the naive
# multi-algorithm mean and Confluence's own consensus baselines (all scored on the
# same rows, per basin and overall).
python src/benchmark_consensus.py \
    --discharge "$WORK_DIR/runs/discharge.csv" \
    --out "$WORK_DIR/runs/scored.csv"

# Evaluate + plot: hydrographs, predicted-vs-observed scatter, per-basin skill
# (NSE / KGE / RMSE), learned-vs-baseline comparison.
python src/evaluate.py \
    --discharge "$WORK_DIR/runs/discharge.csv" \
    --scored "$WORK_DIR/runs/scored.csv" \
    --out-dir "$WORK_DIR/runs/eval"

echo "pipeline complete -> $WORK_DIR/runs/eval"

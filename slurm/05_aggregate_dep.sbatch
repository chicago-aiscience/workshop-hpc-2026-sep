#!/usr/bin/env bash
#SBATCH --job-name=seg-morph
#SBATCH --partition=general          # CPU-only stage; no --gres needed
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/morph_%j.out
# --- Workshop 4: SLURM dependency. Run this ONLY after the inference array
#     finishes, then submit the chain like:
#
#       aid=$(sbatch --parsable slurm/04_infer_array.sbatch)
#       sbatch --dependency=afterok:$aid slurm/05_morphology_dep.sbatch
#
#     `afterok` = start only if every array task succeeded.

set -euo pipefail
source config/paths.sh
source activate "$ENV_NAME"

# Masks (from the array) -> area + circularity features.
python src/morphology.py \
    --mask-dir "$WORK_DIR/runs/masks" \
    --out "$WORK_DIR/runs/morphology.csv"

# Features + survey labels -> supervised quality classifier (grouped by organoid).
python src/classify.py \
    --features "$WORK_DIR/runs/morphology.csv" \
    --labels "$LABELS" \
    --out "$WORK_DIR/runs/classified.csv"

# Evaluate + plot: confusion matrix, balanced accuracy / TNR, feature importance.
python src/evaluate.py \
    --classified "$WORK_DIR/runs/classified.csv" \
    --out-dir "$WORK_DIR/runs/eval"

echo "pipeline complete -> $WORK_DIR/runs/eval"

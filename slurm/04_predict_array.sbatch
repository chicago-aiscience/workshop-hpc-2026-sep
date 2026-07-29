#!/usr/bin/env bash
#SBATCH --job-name=consensus-predict
#SBATCH --partition=general          # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-9                  # 10 parallel tasks (Workshop 4: job arrays)
#SBATCH --output=logs/predict_%A_%a.out   # %A=job id, %a=array index
# --- Workshop 4: predict discharge for ALL reaches in the two basins, in
#     parallel. Each array task handles ~1/10th of the reaches.
#     sbatch slurm/04_predict_array.sbatch
#
#     In production Confluence the algorithm stage fans out one array task per
#     reach/set; here we stride the manifest so the demo stays small.

set -euo pipefail
source config/paths.sh
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"

# Unbuffered stdout/stderr so per-shard progress streams to the .out files live
# (one file per array task); block-buffering would otherwise hide it until exit.
export PYTHONUNBUFFERED=1

# Build the full INFERENCE manifest once (every reach in the basins, no target
# filtering; idempotent).
python src/build_manifest.py \
    --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --basins $BASINS --mode infer \
    --out "$WORK_DIR/data/manifest_infer.csv"

# Each array task predicts discharge for its stride of the manifest (GPU).
python src/predict_discharge.py \
    --manifest "$WORK_DIR/data/manifest_infer.csv" \
    --model "$WORK_DIR/runs/consensus/consensus_model_final.pt" \
    --out-dir "$WORK_DIR/runs/predictions" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT"

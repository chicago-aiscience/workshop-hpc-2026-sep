#!/usr/bin/env bash
#SBATCH --job-name=consensus-predict
#SBATCH --account=pi-dfreedman          # change to an allowed account on the cluster
#SBATCH --partition=schmidt-gpu         # GPU partition for Schmidt fellows
#SBATCH --qos=schmidt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-9                  # 10 parallel tasks (Workshop 4: job arrays)
#SBATCH --output=logs/predict_%A_%a.out   # %A=job id, %a=array index
#SBATCH --error=logs/predict_%A_%a.err
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
    --config "$EXPERIMENT_CONFIG" \
    --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --mode infer \
    --out "$WORK_DIR/data/manifest_infer.csv"

# Coverage: $SLURM_ARRAY_TASK_COUNT is the TOTAL number of array tasks (10 here,
# straight from --array=0-9), so we never hard-code it. predict_discharge strides
# the manifest as rows[index::num_tasks]; tasks 0..N-1 with stride N tile EVERY
# row exactly once -> all reaches get predicted. To scale, change ONLY --array
# (e.g. 0-19) and the count follows. The array MUST start at 0 and be contiguous
# (0-(N-1)) for the tiling to be complete.
python src/predict_discharge.py \
    --manifest "$WORK_DIR/data/manifest_infer.csv" \
    --model "$WORK_DIR/runs/consensus/consensus_model_final.pt" \
    --out-dir "$WORK_DIR/runs/predictions" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --num-tasks "$SLURM_ARRAY_TASK_COUNT"

#!/usr/bin/env bash
#SBATCH --job-name=seg-infer
#SBATCH --partition=general          # <-- your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-9                  # 10 parallel tasks (Workshop 4: job arrays)
#SBATCH --output=logs/infer_%A_%a.out   # %A=job id, %a=array index
# --- Workshop 4: segment 1,000 images in parallel. Each task does ~1/10th.
#     sbatch slurm/04_infer_array.sbatch

set -euo pipefail
source config/paths.sh
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"

# Build the full inference manifest once (raw images, no labels; idempotent).
python src/build_manifest.py --input-dir "$INPUT_DIR" \
    --mode infer --limit 1000 \
    --out "$WORK_DIR/data/manifest_infer.csv"

# Each array task segments its stride of the manifest.
python src/seg_infer.py \
    --manifest "$WORK_DIR/data/manifest_infer.csv" \
    --model "$WORK_DIR/runs/seg/seg_finetune_model_final.pt" \
    --out-dir "$WORK_DIR/runs/masks" \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$SLURM_ARRAY_TASK_COUNT"

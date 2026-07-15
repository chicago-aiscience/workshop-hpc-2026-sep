#!/usr/bin/env bash
#SBATCH --job-name=seg-finetune
#SBATCH --account=pi-dfreedman          # change to an allowed account on the cluster
#SBATCH --partition=schmidt-gpu         # GPU partition for Schmidt fellows
#SBATCH --qos=schmidt
#SBATCH --gres=gpu:1                    # request 1 GPU (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/finetune_%j.out   # stdout  -> normal log  (Workshop 2)
#SBATCH --error=logs/finetune_%j.err    # stderr  -> error log
# --- Workshop 1: a first batch GPU job. Submit with: sbatch slurm/02_finetune.sbatch

set -euo pipefail
source config/paths.sh

# Load required modules
module load python/anaconda-2025.12

# PyTorch from the conda env bundles its own CUDA runtime, so a system CUDA
# module is usually NOT needed (only the node's NVIDIA driver). Load one if your
# cluster has it, but never let a missing module abort the job.
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"           # conda env from environment.yml

# 1. Build the training manifest: raw images paired with MANUAL ground-truth
#    masks (1,000-image subset for the workshop).
python src/build_manifest.py --input-dir "$INPUT_DIR" \
    --mode train --limit 1000 \
    --out "$WORK_DIR/data/manifest_train.csv"

# 2. Fine-tune (a plain fresh run). Preemption-resilient checkpoint/resume is
#    demonstrated separately in slurm/03_checkpoint.sbatch.
python src/seg_finetune.py \
    --manifest "$WORK_DIR/data/manifest_train.csv" \
    --out "$WORK_DIR/runs/seg" \
    --epochs 10 --batch-size 4

echo "finetune complete on $(hostname) job $SLURM_JOB_ID"

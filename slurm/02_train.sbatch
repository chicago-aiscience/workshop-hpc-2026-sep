#!/usr/bin/env bash
#SBATCH --job-name=consensus-train
#SBATCH --account=pi-dfreedman          # change to an allowed account on the cluster
#SBATCH --partition=schmidt-gpu         # GPU partition for Schmidt fellows
#SBATCH --qos=schmidt
#SBATCH --gres=gpu:1                    # request 1 GPU (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out      # stdout  -> normal log  (Workshop 2)
#SBATCH --error=logs/train_%j.err       # stderr  -> error log
# --- Workshop 1: a first batch GPU job. Submit with: sbatch slurm/02_train.sbatch

set -euo pipefail
source config/paths.sh

# Load required modules
module load python/anaconda-2025.12

# PyTorch from the conda env bundles its own CUDA runtime, so a system CUDA
# module is usually NOT needed (only the node's NVIDIA driver). Load one if your
# cluster has it, but never let a missing module abort the job.
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"           # conda env from environment.yml

# Unbuffered stdout/stderr so print() and logging stream to the .out/.err files
# live (block-buffering would otherwise hide progress until the buffer fills or
# the job ends). Covers all python steps below, so no per-call flush needed.
export PYTHONUNBUFFERED=1

# 1. Build the TRAINING manifest: one row per (reach, SWOT overpass) for the two
#    workshop basins, joining the FLPE algorithm discharges + ML-prior features
#    with the observed gauge discharge TARGET (kept only where a gauge exists).
python src/build_manifest.py \
    --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --basins $BASINS --mode train \
    --out "$WORK_DIR/data/manifest_train.csv"

# 2. Train the learned-consensus MLP (a plain fresh run). Preemption-resilient
#    checkpoint/resume is demonstrated separately in slurm/03_checkpoint.sbatch.
python src/train_consensus.py \
    --manifest "$WORK_DIR/data/manifest_train.csv" \
    --out "$WORK_DIR/runs/consensus" \
    --epochs 500 --batch-size 256

echo "training complete on $(hostname) job $SLURM_JOB_ID"

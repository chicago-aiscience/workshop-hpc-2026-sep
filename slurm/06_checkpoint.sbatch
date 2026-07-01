#!/usr/bin/env bash
#SBATCH --job-name=seg-ckpt
#SBATCH --partition=general          # use a preemptible partition to demo this
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue                    # let SLURM put the job back in the queue...
#SBATCH --signal=B:USR1@120          # ...and warn us 120s before it's killed
#SBATCH --output=logs/ckpt_%j.out
# --- Workshop 2 + 5: surviving preemption via checkpoint/resume.
#
#     The training script saves runs/seg/last.pt every epoch and reloads it with
#     --resume. If this job is preempted and requeued, it continues from the last
#     checkpoint instead of starting over. Submit normally:
#         sbatch slurm/06_checkpoint.sbatch
#     ...then cancel+requeue (or trigger preemption) and watch it resume.

set -euo pipefail
source config/paths.sh
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"

# Optional: react to the pre-emption warning signal for a clean last save.
trap 'echo "got USR1 — checkpoint already on disk, exiting cleanly"; exit 0' USR1

python src/seg_finetune.py \
    --manifest "$WORK_DIR/data/manifest_train.csv" \
    --out "$WORK_DIR/runs/seg" \
    --epochs 50 --batch-size 4 --resume &
wait                                 # `wait` lets the trap fire during the run

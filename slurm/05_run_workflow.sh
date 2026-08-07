#!/usr/bin/env bash
# --- Lesson 4: submit the whole three-stage workflow in one go.
#
#       bash slurm/05_run_workflow.sh
#
#     SAFE TO RUN ON A LOGIN NODE. This script does no computation - it calls
#     sbatch three times and exits in well under a second. All the real work
#     happens on compute nodes, under Slurm's control.
#
#     Stage 1  02_train.sbatch          single GPU job
#     Stage 2  04_predict_array.sbatch  job array, released on afterok of stage 1
#     Stage 3  05_aggregate_dep.sbatch  single CPU job, released on afterok of stage 2

set -euo pipefail

# sbatch records the directory you submit from as each job's working directory,
# and the #SBATCH --output paths and `source config/paths.sh` lines in those
# scripts are all relative to the repo root. Resolve the root from this script's
# own location so the chain submits correctly no matter where you call it from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# The --output/--error paths point into logs/; Slurm rejects the submission
# outright if that directory doesn't exist yet.
mkdir -p logs

train_id=$(sbatch --parsable slurm/02_train.sbatch)
echo "stage 1  train      -> job ${train_id}"

# afterok on an array job waits for EVERY task in that array to succeed.
predict_id=$(sbatch --parsable --dependency=afterok:"${train_id}" slurm/04_predict_array.sbatch)
echo "stage 2  predict    -> job ${predict_id}  (held until ${train_id} succeeds)"

aggregate_id=$(sbatch --parsable --dependency=afterok:"${predict_id}" slurm/05_aggregate_dep.sbatch)
echo "stage 3  aggregate  -> job ${aggregate_id}  (held until ${predict_id} succeeds)"

# If a stage fails, its dependents do not clear on their own - they sit in PD
# with reason DependencyNeverSatisfied until cancelled.
echo
echo "watch:  squeue -u \$USER"
echo "cancel: scancel ${train_id} ${predict_id} ${aggregate_id}"

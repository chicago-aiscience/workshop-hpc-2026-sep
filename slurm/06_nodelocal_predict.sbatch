#!/usr/bin/env bash
#SBATCH --job-name=consensus-nodelocal
#SBATCH --account=pi-dfreedman          # change to an allowed account on the cluster
#SBATCH --partition=general             # <-- your CPU partition (add --gres=gpu:1 to predict on GPU)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nodelocal_%j.out
#SBATCH --error=logs/nodelocal_%j.err
# --- Lesson 3 (Data management): stage inputs to fast NODE-LOCAL disk, do the
#     predict -> aggregate work there, then copy ONLY the aggregated result back
#     to shared storage.
#
#       sbatch slurm/06_nodelocal_predict.sbatch
#
#     Why: reading the netCDF inputs and writing the many small per-shard
#     prediction files on node-local disk keeps that traffic OFF the shared
#     filesystem (the many-small-files problem from Lesson 3). Only the single
#     rolled-up discharge.csv is worth keeping, so only that is copied back.
#
#     Needs a trained model at $WORK_DIR/runs/consensus/consensus_model_final.pt
#     (from slurm/02_train.sbatch or 03_checkpoint.sbatch). Create logs/ first:
#       mkdir -p logs

set -euo pipefail
source config/paths.sh
source activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

NSHARDS=10                                                   # prediction shards (looped on one node)
MODEL="$WORK_DIR/runs/consensus/consensus_model_final.pt"

# --- 1. Pick a node-local scratch dir -------------------------------------------
# $SLURM_TMPDIR is per-job disk physically ON the compute node: fast and private,
# but EPHEMERAL -- Slurm wipes it at job end. Fall back to a job-unique /tmp path
# if the site doesn't set it.
LOCAL="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL/input" "$LOCAL/predictions"

# Always clean up node-local scratch on exit (success, error, or signal) so we
# never strand data on the compute node.
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT

echo "[nodelocal] staging inputs to $LOCAL on $(hostname)"

# --- 2. Stage inputs: shared filesystem -> node-local ---------------------------
# Copy the read-only netCDFs and the trained model onto local disk ONCE, so the
# compute reads from fast local storage instead of repeatedly hitting the shared FS.
cp "$SOS_FILE" "$SVS_FILE" "$LOCAL/input/"
cp "$MODEL" "$LOCAL/model.pt"
SOS_LOCAL="$LOCAL/input/$(basename "$SOS_FILE")"
SVS_LOCAL="$LOCAL/input/$(basename "$SVS_FILE")"

# Priors are optional: stage + pass them only if the file exists.
PRIORS_ARG=()
if [[ -f "$PRIORS_FILE" ]]; then
    cp "$PRIORS_FILE" "$LOCAL/input/"
    PRIORS_ARG=(--priors "$LOCAL/input/$(basename "$PRIORS_FILE")")
fi

# --- 3. Build the inference manifest on node-local ------------------------------
python src/build_manifest.py \
    --config "$EXPERIMENT_CONFIG" \
    --sos "$SOS_LOCAL" --svs "$SVS_LOCAL" "${PRIORS_ARG[@]}" \
    --mode infer --out "$LOCAL/manifest_infer.csv"

# --- 4. Predict every shard, writing the many small CSVs to LOCAL disk ----------
# On the real cluster these shards fan out across a job array (04_predict_array);
# here we loop them on one node so all the small intermediate files land on fast
# local disk rather than the shared filesystem.
for ((i = 0; i < NSHARDS; i++)); do
    python src/predict_discharge.py \
        --manifest "$LOCAL/manifest_infer.csv" \
        --model "$LOCAL/model.pt" \
        --out-dir "$LOCAL/predictions" \
        --index "$i" --num-tasks "$NSHARDS"
done

# --- 5. Aggregate on node-local -------------------------------------------------
# Roll the many per-shard CSVs up into ONE table, still on local disk.
python src/aggregate_discharge.py \
    --pred-dir "$LOCAL/predictions" \
    --out "$LOCAL/discharge.csv"

# --- 6. Move ONLY the aggregated result back to shared storage ------------------
# This single file is what we keep; the per-shard files stay on node-local and are
# wiped by the trap. Copy BEFORE the job ends -- node-local disk does not persist.
mkdir -p "$WORK_DIR/runs"
cp "$LOCAL/discharge.csv" "$WORK_DIR/runs/discharge.csv"
echo "[nodelocal] wrote $WORK_DIR/runs/discharge.csv ($(wc -l < "$WORK_DIR/runs/discharge.csv") lines)"

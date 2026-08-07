#!/usr/bin/env bash
#SBATCH --job-name=consensus-nodelocal
#SBATCH --account=pi-dfreedman          # RCC: your Slurm allocation (change to one you can use)
#SBATCH --partition=caslake             # RCC CPU partition (GPU: --partition=schmidt-gpu --qos=schmidt --gres=gpu:1)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/nodelocal_%j.out
#SBATCH --error=logs/nodelocal_%j.err
# --- Lesson 2 (Data management): stage inputs to fast NODE-LOCAL disk, do the
#     predict -> aggregate work there, then copy ONLY the aggregated result back
#     to shared storage.
#
#       sbatch slurm/06_nodelocal_predict.sbatch
#
#     Why: reading the netCDF inputs and writing the many small per-shard
#     prediction files on node-local disk keeps that traffic OFF the shared
#     filesystem (the many-small-files problem from Lesson 2). Only the single
#     rolled-up discharge.csv is worth keeping, so only that is copied back.
#
#     Needs a trained model at $WORK_DIR/runs/consensus/consensus_model_final.pt
#     (from slurm/02_train.sbatch or 03_checkpoint.sbatch). Create logs/ first:
#       mkdir -p logs
#
#     -------------------------------------------------------------------------
#     This script defaults to RCC (Midway). To run it on the DSI cluster, make
#     three changes (RCC needs none of them -- see Lesson 2, node-local sections):
#       1. Partition: use  #SBATCH --partition=general  and drop the RCC-only
#          --qos=schmidt / schmidt-gpu settings (add --gres=gpu:1 for a GPU node).
#       2. Request node-local disk: add  #SBATCH --gres=local:50G  (size to fit
#          your inputs). RCC hands every job node-local scratch automatically;
#          DSI only places you on a node with local disk if you REQUEST it.
#       3. Node-local path: in step 1 below, set
#             LOCAL=/local/scratch/${USER}_${SLURM_JOB_ID}
#          DSI does not set $SLURM_TMPDIR, so the fallback would land on slow
#          shared /tmp instead of the fast NVMe local disk.
#     -------------------------------------------------------------------------

set -euo pipefail
source config/paths.sh
source activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

NSHARDS=10                                                   # prediction shards (looped on one node)
MODEL="$WORK_DIR/runs/consensus/consensus_model_final.pt"

# --- 1. Pick a node-local scratch dir -------------------------------------------
# RCC: $SLURM_TMPDIR is the per-job disk physically ON the compute node (fast and
# private, but EPHEMERAL -- Slurm wipes it at job end); every RCC job gets one for
# free. The /tmp fallback just covers a site that doesn't set the variable.
# DSI: $SLURM_TMPDIR is NOT set -- change this to
#   LOCAL=/local/scratch/${USER}_${SLURM_JOB_ID}
# and request it with --gres=local (see the header). Otherwise the fallback lands
# on slow shared /tmp, not the fast local disk.
LOCAL="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
mkdir -p "$LOCAL/input" "$LOCAL/predictions"

# Always clean up node-local scratch on exit (success, error, or signal) so we
# never strand data on the compute node.
cleanup() { rm -rf "$LOCAL"; }
trap cleanup EXIT

echo "[nodelocal] staging inputs to $LOCAL on $(hostname)"

# --- 2. Stage inputs: shared filesystem -> node-local ---------------------------
# rsync the read-only netCDFs and the trained model onto local disk ONCE, so the
# compute reads from fast local storage instead of repeatedly hitting the shared FS.
# rsync -a preserves timestamps/permissions and only transfers what's missing, so
# re-running (e.g. after a requeue) skips files already staged.
rsync -a "$SOS_FILE" "$SVS_FILE" "$LOCAL/input/"
rsync -a "$MODEL" "$LOCAL/model.pt"
SOS_LOCAL="$LOCAL/input/$(basename "$SOS_FILE")"
SVS_LOCAL="$LOCAL/input/$(basename "$SVS_FILE")"

# Priors are optional: stage + pass them only if the file exists.
PRIORS_ARG=()
if [[ -f "$PRIORS_FILE" ]]; then
    rsync -a "$PRIORS_FILE" "$LOCAL/input/"
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
# wiped by the trap. rsync it back BEFORE the job ends -- node-local disk does not
# persist. rsync over cp gives an atomic-ish, resumable transfer of the result.
mkdir -p "$WORK_DIR/runs"
rsync -a "$LOCAL/discharge.csv" "$WORK_DIR/runs/discharge.csv"
echo "[nodelocal] wrote $WORK_DIR/runs/discharge.csv ($(wc -l < "$WORK_DIR/runs/discharge.csv") lines)"

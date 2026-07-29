#!/usr/bin/env bash
#SBATCH --job-name=consensus-ckpt
#SBATCH --partition=general          # use a preemptible partition to demo this
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue                    # allow SLURM to put the job back in the queue
#SBATCH --signal=B:USR2@120          # warn the batch shell (USR2) 120s before the kill
#SBATCH --output=logs/ckpt_%j.out
# --- Lesson 3: surviving interruptions via checkpoint/resume + signal handling.
#
#     This is meant as a general pattern for cluster work. Clusters differ: some
#     PREEMPT jobs (bump a running job for a higher-priority one), others do not
#     (e.g. RCC/midway has preemption disabled), and every cluster enforces a
#     --time wall. The same machinery covers all of these:
#       - Time-limit wall  -> SLURM sends the SIGUSR2 warning (see --signal below)
#                             ~120s early; we checkpoint and requeue to continue.
#       - Preemption        -> on clusters that enable it, handled the same way
#                             (warning signal) or by SLURM's automatic requeue.
#       - Node failure      -> SLURM auto-requeues (--requeue); resume from the
#                             last interval/epoch checkpoint.
#
#     src/train_consensus.py saves a rolling checkpoint (consensus_last.pt) on a
#     regular interval, at each epoch boundary, AND when it receives the SIGUSR2
#     warning -- then exits cleanly. This wrapper catches the same warning, waits
#     for the checkpoint, and requeues the job with `scontrol requeue`. On restart
#     it hands the checkpoint back with --resume-from and training continues.
#     Submit normally:
#         sbatch slurm/03_checkpoint.sbatch
#     ...then send USR2 yourself (or hit the --time wall) and watch it requeue.
#
#     Signal choice: USR2 (not USR1) because USR1 is already used by some
#     frameworks (PyTorch Lightning, submitit) for their own auto-requeue.
#
#     Wrapper pattern adapted from the UChicago DSI cluster batch-jobs guide
#     (which uses an exit-99 requeue convention; we use the portable
#     `scontrol requeue` instead -- see the note near the bottom):
#     https://cluster-policy.ds.uchicago.edu/using-the-cluster/batch-jobs/

set -euo pipefail  # prevent silent failures

# Load and activate software dependencies
source config/paths.sh
module load cuda 2>/dev/null || echo "[job] no 'cuda' module; using conda torch's bundled CUDA"
source activate "$ENV_NAME"

# Unbuffered stdout/stderr so print() and logging stream to the .out/.err files
# live -- important here so checkpoint/preemption messages appear immediately
# rather than being trapped in a buffer when the job is signalled.
export PYTHONUNBUFFERED=1

# --- Tunables ---------------------------------------------------------------
CKPT_DIR="$WORK_DIR/runs/consensus"  # where the rolling consensus_last.pt lives
MAX_RESTARTS=5                       # stop requeuing after this many restarts
# Model + training knobs (incl. checkpoint_interval) live in $EXPERIMENT_CONFIG;
# edit config/experiments/baseline.yaml to change them (or pass a CLI flag above).
# ----------------------------------------------------------------------------

mkdir -p "$CKPT_DIR"

# Build the training manifest if a previous stage hasn't already (idempotent),
# so this job can also be submitted on its own.
if [[ ! -f "$WORK_DIR/data/manifest_train.csv" ]]; then
    python src/build_manifest.py \
        --config "$EXPERIMENT_CONFIG" \
        --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
        --mode train \
        --out "$WORK_DIR/data/manifest_train.csv"
fi

# Guard against an infinite requeue loop: if the job keeps dying before making
# progress, give up instead of resubmitting forever.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt "$MAX_RESTARTS" ]]; then
    echo "[$(date)] restart count ${SLURM_RESTART_COUNT} exceeds MAX_RESTARTS=${MAX_RESTARTS}; giving up."
    exit 1
fi

child_pid=""

# Forward a signal to the whole training process group, falling back to the
# direct child PID. This relies on launching Python with `setsid` below: the
# child PID is then also the process-group ID, so `-$child_pid` targets the whole
# group. That matters for parallel workers (PyTorch DataLoader, multiprocessing,
# joblib, ...) which would otherwise be orphaned.
signal_child_group() {
    local sig="$1"
    if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
        echo "[$(date)] Sending ${sig} to training process group..."
        kill -s "${sig}" -- "-$child_pid" 2>/dev/null || \
            kill -s "${sig}" "$child_pid" 2>/dev/null || \
            true
    fi
}

# SIGUSR2 warning handler. This is the signal SLURM sends ~120s before the --time
# wall (see --signal above); on clusters that enable preemption and are configured
# to send a warning, the same signal fires then too. Tell Python to checkpoint,
# wait for it, then requeue the job ourselves with `scontrol requeue` (portable;
# see the note near the bottom). `trap ''` first so the signals we're about to
# provoke (scontrol requeue terminates this run) don't re-enter the handlers.
on_preempt() {
    trap '' USR2 TERM
    echo "[$(date)] USR2 received: signalling Python to checkpoint before time-limit/preemption."
    signal_child_group USR2
    wait "$child_pid" || true          # let Python checkpoint; grace window is the hard deadline
    echo "[$(date)] Requeuing job ${SLURM_JOB_ID} via scontrol requeue."
    scontrol requeue "$SLURM_JOB_ID" || \
        echo "[$(date)] scontrol requeue failed (needs --requeue and job ownership)."
    exit 0
}

# TERM reaching the batch shell (e.g. scancel --batch): terminate without an
# intentional requeue. Plain `scancel <jobid>` may signal Python directly instead,
# which the `wait` below then observes.
on_term() {
    trap '' USR2 TERM
    echo "[$(date)] TERM received: terminating without intentional requeue."
    signal_child_group TERM
    wait "$child_pid" || true
    echo "[$(date)] Exiting 143 (128 + SIGTERM)."
    exit 143
}

trap on_preempt USR2
trap on_term TERM

# ============================================================================
# >>> EDIT HERE: launch YOUR training program <<<
#
# This is the only part you change to reuse this wrapper for your own code.
# Swap in your script and its arguments below. To work with the wrapper, your
# program MUST honor this contract:
#   1. Catch SIGUSR2 -> checkpoint at a safe point -> exit cleanly.
#      (The wrapper does the requeue; your program just saves and exits.)
#   2. Resume from the checkpoint path passed in $1 (empty string => start fresh).
#   3. Write its checkpoint(s) under $CKPT_DIR so the next run can find them.
# See src/train_consensus.py (handle_preempt / atomic_torch_save / --resume-from)
# for a reference implementation.
#
# Keep the `setsid ... &` form: `setsid` makes the program its own process-group
# leader (so signal_child_group can reach its workers) and `&` backgrounds it so
# the traps can fire while it runs.
# ============================================================================
launch_training() {
    local resume_from="$1"
    local resume_args=()
    [[ -n "$resume_from" ]] && resume_args=(--resume-from "$resume_from")

    setsid python src/train_consensus.py \
        --config "$EXPERIMENT_CONFIG" \
        --manifest "$WORK_DIR/data/manifest_train.csv" \
        --out "$WORK_DIR/runs/consensus" \
        --ckpt-dir "$CKPT_DIR" \
        "${resume_args[@]}" &
}
# ============================================================================

# Resume from the rolling checkpoint if one exists (a previous run left it here).
latest_ckpt="$CKPT_DIR/consensus_last.pt"
if [[ -f "$latest_ckpt" ]]; then
    echo "[$(date)] Resuming from $latest_ckpt (restart #${SLURM_RESTART_COUNT:-0})"
else
    echo "[$(date)] No checkpoint found; starting a fresh training run."
    latest_ckpt=""
fi

launch_training "$latest_ckpt"
child_pid=$!    # PID of the setsid'd process backgrounded inside launch_training

# If we get here, training ran to completion (or crashed) WITHOUT a preemption
# warning -- the on_preempt handler above never fired, so no requeue is wanted.
return_code=0
wait "$child_pid" || return_code=$?
echo "[$(date)] Training exited with code ${return_code}"

# NOTE ON REQUEUE (portability): this wrapper requeues explicitly with
# `scontrol requeue` inside on_preempt. That works on ANY SLURM cluster as long as
# the job is submitted with `--requeue` and you own the job -- no special cluster
# configuration required. Some sites instead wire an `exit 99` convention to
# SLURM's `RequeueExit` (e.g. the DSI cluster: "exit 99 tells Slurm to requeue"),
# but that is cluster-specific, so we avoid depending on it here. Separately,
# `--requeue` on its own already auto-requeues on preemption / node failure.

exit "$return_code"

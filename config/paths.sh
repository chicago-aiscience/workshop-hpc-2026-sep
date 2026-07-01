#!/usr/bin/env bash
# Shared paths for all SLURM scripts. `source config/paths.sh` at the top of a job.
#
# These are set UNCONDITIONALLY (plain `export`, not `${VAR:=...}`) on purpose:
# SLURM copies the submitting shell's environment into jobs (--export=ALL), so a
# stale or inherited value would otherwise shadow the default. Unconditional
# assignment means this file always wins -- to relocate the project, edit the
# two paths below. (Per-run overrides: pass --input-dir / --labels to the
# scripts directly.)

# Raw inputs: images / masks / json / labels.csv.
export INPUT_DIR=/data/workshop-hpc-data/workshop-hpc-data-mini/input

# Where jobs write manifests, checkpoints, masks, results (scratch, NOT $HOME).
export WORK_DIR=/data/workshop-hpc-data/workshop-hpc-data-mini/output

# Survey labels CSV, derived from INPUT_DIR so it always tracks the data.
export LABELS="$INPUT_DIR/labels.csv"

# Conda/micromamba env (a name, or a full prefix path for a -p env).
export ENV_NAME=workshop-hpc

mkdir -p "$WORK_DIR/data" "$WORK_DIR/runs"

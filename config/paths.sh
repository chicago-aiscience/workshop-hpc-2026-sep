#!/usr/bin/env bash
# Shared paths for all SLURM scripts. `source config/paths.sh` at the top of a job.
#
# These are set UNCONDITIONALLY (plain `export`, not `${VAR:=...}`) on purpose:
# SLURM copies the submitting shell's environment into jobs (--export=ALL), so a
# stale or inherited value would otherwise shadow the default. Unconditional
# assignment means this file always wins -- to relocate the project, edit the
# paths below. (Per-run overrides: pass --sos / --priors / --basins to the
# scripts directly.)

# Read-only project inputs: the workshop mini-dataset (tens of MB), produced once
# by data_prep/build_mini_dataset.py from the full public SWOT Confluence products.
export INPUT_DIR=/data/workshop-hpc-data/confluence-mini/input

# SoS results: per-reach FLPE algorithm discharge (metroman, momma, neobam,
# sic4dvar) + Confluence consensus, sliced to the two workshop basins.
export SOS_FILE="$INPUT_DIR/eu_SOS_mini.nc"

# SWOT Validation Set: observed daily gauge discharge, keyed by reach_id (v16) --
# the supervised training TARGET.
export SVS_FILE="$INPUT_DIR/svs_mini.nc"

# ML-prior flow statistics per reach (mean_q, monthly_q) used as extra features.
export PRIORS_FILE="$INPUT_DIR/priors_mini.csv"

# Experiment config: model + training knobs AND data.basins, tracked in Git.
# Copy config/experiments/baseline.yaml to define a new experiment. Path is
# relative to the repo root, where jobs run (same as the src/... paths the
# scripts call). Per-run override: pass --basins / --epochs / etc. to a script.
export EXPERIMENT_CONFIG="config/experiments/baseline.yaml"

# Where jobs write manifests, checkpoints, predictions, results (scratch, NOT $HOME).
export WORK_DIR=/data/workshop-hpc-data/confluence-mini/output

# Conda/micromamba env (a name, or a full prefix path for a -p env).
export ENV_NAME=workshop-hpc

mkdir -p "$WORK_DIR/data" "$WORK_DIR/runs"

#!/usr/bin/env bash
# Create (or rebuild) the workshop conda env in scratch space.
#
# Scratch is periodically PURGED, so this script is idempotent: keep it and
# environment.yml in durable space (the repo), and re-run it whenever the env is
# gone and it rebuilds. If a valid env is already present it does nothing
# (pass --force to rebuild anyway).
#
#   ./setup_env.sh            # build if missing, else skip
#   ./setup_env.sh --force    # remove and rebuild from scratch
#
# Locations default to per-user scratch; override by exporting before running:
#   ENV_PREFIX=/some/path ./setup_env.sh
set -euo pipefail

# --- config (per-user scratch by default) ---
SCRATCH_BASE="${SCRATCH_BASE:-/net/scratch2/$USER}"
ENV_PREFIX="${ENV_PREFIX:-$SCRATCH_BASE/workshop-hpc-env}"
PKGS_DIR="${CONDA_PKGS_DIRS:-$SCRATCH_BASE/conda-pkgs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/environment.yml}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# --- pick whichever package manager is available (mamba is fastest) ---
if command -v micromamba >/dev/null 2>&1; then MGR=micromamba
elif command -v mamba    >/dev/null 2>&1; then MGR=mamba
elif command -v conda    >/dev/null 2>&1; then MGR=conda
else echo "ERROR: no conda / mamba / micromamba on PATH" >&2; exit 1
fi

echo "[setup] manager = $MGR"
echo "[setup] env     = $ENV_PREFIX"
echo "[setup] pkgs    = $PKGS_DIR"
echo "[setup] spec    = $ENV_FILE"
[ -f "$ENV_FILE" ] || { echo "ERROR: environment.yml not found at $ENV_FILE" >&2; exit 1; }

# Package cache on scratch too: the read-only system conda often can't write to
# its own /opt/conda/pkgs (the PermissionError you hit).
export CONDA_PKGS_DIRS="$PKGS_DIR"
mkdir -p "$PKGS_DIR"

# --- build only if missing/broken (or forced) ---
if [ "$FORCE" -eq 0 ] && [ -x "$ENV_PREFIX/bin/python" ]; then
    echo "[setup] valid env already present -> skipping build (use --force to rebuild)"
else
    [ -e "$ENV_PREFIX" ] && { echo "[setup] removing existing/partial env"; rm -rf "$ENV_PREFIX"; }
    echo "[setup] building env (torch download can take several minutes)..."
    case "$MGR" in
        micromamba) micromamba create -y -p "$ENV_PREFIX" -f "$ENV_FILE" ;;
        mamba)      mamba env create   -p "$ENV_PREFIX" -f "$ENV_FILE" ;;
        conda)      conda env create   -p "$ENV_PREFIX" -f "$ENV_FILE" ;;
    esac
fi

# --- verify (cuda is expectedly False on a login node; True on a GPU node) ---
echo "[setup] verifying torch import..."
"$MGR" run -p "$ENV_PREFIX" python - <<'PY'
import torch
print(f"  torch {torch.__version__} | cuda build {torch.version.cuda} | "
      f"cuda available: {torch.cuda.is_available()}")
PY

echo
echo "[setup] done. Activate with:"
echo "    conda activate $ENV_PREFIX        # or: micromamba activate $ENV_PREFIX"
echo "[setup] tip: set ENV_NAME=$ENV_PREFIX in config/paths.sh so jobs use it."

# Lesson 1 — Interactive session walkthrough

Before submitting batch jobs, get a feel for the cluster live. This is the
"hello world" of the project: log in, grab a GPU interactively, train the model
on *one* basin.

## 1. Log into the cluster

```bash
ssh -Y <CNetID>@midway3.rcc.uchicago.edu
```

Enter your `<CNETID>` password and then authenticate with Duo.

## 2. Clone the GitHub repository to the cluster

Note this **requires SSH authentication** be set up between GitHub and the cluster. See [SSH + GitHub](appendix.md#ssh-github)

*Start the SSH agent and load the SSH key you created:*

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

*Clone the repository:*

```bash
git clone git@github.com:chicago-aiscience/workshop-hpc-2026-sep.git
cd workshop-hpc-2026-sep
```

## 3. Set up the environment (once)

*Run the following command to set up the `conda` environment in your scratch space:*

```bash
SCRATCH_BASE=/scratch/midway3/ntebaldi bash config/setup_env.sh  # run the script (bash executable)
```

**4. Grab an interactive GPU**

```bash
srun --account pi-dfreedman \
     --partition schmidt-gpu \
     --qos=schmidt \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=2 \
     --mem=4GB \
     --gres=gpu:1 \
     --time=01:00:00 \
     --pty bash -i
```

`--pty bash` drops you into a shell *on the compute node*. Confirm the GPU:

```bash
nvidia-smi          # should list one GPU
```

## 4. Copy the data to the cluster

This is a quick preview of the data storage lesson which we will cover in more detail. For now, download the input data needed to run the workshop repository code to your laptop: https://drive.google.com/drive/folders/1kGueYViLp8cWjQhKCXsUP2wbJIt1BrBd?usp=share_link

The workshop mini-dataset is small (~5 MB) — the SWOT Confluence products sliced to two river basins (Loire + Rhine) — so let's just copy it via `scp`. It is three files: `eu_SOS_mini.nc` (algorithm discharge), `svs_mini.nc` (gauge observations), and `priors_mini.csv` (ML-prior features):

```bash
mkdir /scratch/midway3/$USER/workshop-hpc-data/  # create a directory to hold the data

scp -r /path/to/input/directory/on/your/laptop midway:/scratch/midway3/$USER/workshop-hpc-data  # copy the data to the cluster
```

## 5. Define the data paths on the cluster

Modify `config/paths.sh` for the RCC cluster:

```bash
#!/usr/bin/env bash

# Read-only project inputs: the workshop mini-dataset.
export INPUT_DIR=/scratch/midway3/$USER/workshop-hpc-data/input

# SoS results (FLPE algorithm discharge + Confluence consensus).
export SOS_FILE="$INPUT_DIR/eu_SOS_mini.nc"

# SWOT Validation Set: observed daily gauge discharge = the training TARGET.
export SVS_FILE="$INPUT_DIR/svs_mini.nc"

# ML-prior flow statistics per reach (extra features).
export PRIORS_FILE="$INPUT_DIR/priors_mini.csv"

# Experiment config: model + training knobs AND data.basins, tracked in Git.
export EXPERIMENT_CONFIG="config/experiments/baseline.yaml"

# Where jobs write manifests, checkpoints, predictions, results (scratch, NOT $HOME).
export WORK_DIR=/scratch/midway3/$USER/workshop-hpc-data/output

# Conda/micromamba env (a name, or a full prefix path for a -p env).
export ENV_NAME=/scratch/midway3/$USER/workshop-hpc-data/workshop-hpc-env

mkdir -p "$WORK_DIR/data" "$WORK_DIR/runs"
```

## 6. Train the model on one basin

```bash
# Load the paths you defined
source config/paths.sh

# Build a tiny training manifest for a single sub-basin (232270 = upper Loire):
# one row per (reach, SWOT overpass) with the algorithm discharges + gauge target.
# The experiment config supplies data.basins; --basins here OVERRIDES it to just
# the one sub-basin for a fast smoke test.
python src/build_manifest.py \
    --config "$EXPERIMENT_CONFIG" \
    --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --basins 232270 --mode train --out "$WORK_DIR"/tiny.csv

# Train the learned-consensus model for a single epoch as a smoke test.
# The config supplies epochs/batch_size/lr/hidden; --epochs 1 overrides it here.
python src/train_consensus.py \
    --config "$EXPERIMENT_CONFIG" \
    --manifest "$WORK_DIR"/tiny.csv \
    --out "$WORK_DIR"/consensus --epochs 1
```

Watch the loss print. When it finishes, `exit` releases the node. The exact knobs
this run used are recorded at `"$WORK_DIR"/consensus/config_used.yaml`.

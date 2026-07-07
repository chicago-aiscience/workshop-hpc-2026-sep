# Workshop 1 — Interactive session walkthrough

Before submitting batch jobs, get a feel for the cluster live. This is the
"hello world" of the project: log in, grab a GPU interactively, run the model on
*one* image.

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

The data is on the smaller side (~61M) so lets just copy it via `scp`:

```bash
mkdir /scratch/midway3/$USER/workshop-hpc-data/  # create a directory to hold the data

scp -r /path/to/input/directory/on/your/laptop midway:/scratch/midway3/$USER/workshop-hpc-data  # copy the data to the cluster
```

## 5. Define the data paths on the cluster

Modify `config/paths.sh` for the RCC cluster:

```bash
#!/usr/bin/env bash

# Raw inputs: images / masks / json / labels.csv.
export INPUT_DIR=/scratch/midway3/$USER/workshop-hpc-data/input

# Where jobs write manifests, checkpoints, masks, results (scratch, NOT $HOME).
export WORK_DIR=/scratch/midway3/$USER/workshop-hpc-data/output

# Survey labels CSV, derived from INPUT_DIR so it always tracks the data.
export LABELS="$INPUT_DIR/labels.csv"

# Conda/micromamba env (a name, or a full prefix path for a -p env).
export ENV_NAME=/scratch/midway3/$USER/workshop-hpc-data/workshop-hpc-env

mkdir -p "$WORK_DIR/data" "$WORK_DIR/runs"
```

## 6. Run the model on one image

```bash
# Load the paths you defined
source config/paths.sh

# Build the image/mask pairs manifest
python src/build_manifest.py --input-dir "$INPUT_DIR" --mode train --limit 5 \
    --out "$WORK_DIR"/tiny.csv

# Fine tune the image segmentation model
python src/seg_finetune.py --manifest "$WORK_DIR"/tiny.csv\
    --out "$WORK_DIR"/seg --epochs 1
```

Watch the loss print. When it finishes, `exit` releases the node.

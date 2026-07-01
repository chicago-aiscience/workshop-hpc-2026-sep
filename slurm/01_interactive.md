# Workshop 1 — Interactive session walkthrough

Before submitting batch jobs, get a feel for the cluster live. This is the
"hello world" of the project: log in, grab a GPU interactively, run the model on
*one* image.

## 1. Log in (DSI cluster)

You need an SSH key on file first — see
<https://cluster-policy.ds.uchicago.edu/using-the-cluster/login-nodes/>.

```bash
ssh <cnetid>@fe.ds.uchicago.edu        # front-end / login node
```

> Do **not** run compute on the login node — it's shared. Always `srun`/`sbatch`.

## 2. Set up the environment (once)

```bash
cd ~/workshop-hpc                       # this project, copied to the cluster
module avail                            # what's installed? (note exact CUDA name, if any)
# Optional: PyTorch from the env bundles its own CUDA runtime, so a system CUDA
# module usually isn't needed -- only the node's NVIDIA driver. Skip if absent.
module load cuda 2>/dev/null || echo "no cuda module; conda torch brings its own"
conda env create -f environment.yml     # build the pinned env
conda activate workshop-hpc
```

## 3. Grab an interactive GPU node

```bash
srun --partition=general --gres=gpu:1 --cpus-per-task=4 \
     --mem=16G --time=00:30:00 --pty bash
```

`--pty bash` drops you into a shell *on the compute node*. Confirm the GPU:

```bash
nvidia-smi          # should list one GPU
```

## 4. Run the model on one image

```bash
source config/paths.sh
python src/build_manifest.py --input-dir "$INPUT_DIR" --mode train --limit 5 \
    --out /tmp/tiny.csv
python src/seg_finetune.py --manifest /tmp/tiny.csv --out /tmp/seg --epochs 1
```

Watch the loss print. When it finishes, `exit` releases the node.

## Concepts introduced here

- **Login vs. compute nodes** — never compute on the front-end.
- **Modules** (`module load cuda`) vs. **conda env** (Python packages).
- **Partitions** (`--partition`) and **interactive jobs** (`srun --pty`).
- Next: turn this into an unattended **batch** job → `02_finetune.sbatch`.

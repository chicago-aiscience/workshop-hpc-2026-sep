# Workshop Guide — mapping the project to the 5 sessions

One project, used end to end. Each session adds one layer of cluster skill on
top of the same segmentation task.

## 1. Getting Started
**Goal:** log in, environment, first job.
- `slurm/01_interactive.md` — SSH to DSI (login vs. compute nodes), `module load`,
  `conda env create -f environment.yml`, grab a GPU with `srun --pty`, run the
  model on one image.
- `environment.yml` — modules vs. conda environments.
- `slurm/02_finetune.sbatch` — turn the interactive run into a submitted batch
  job; partitions, `--gres=gpu:1`, `sbatch`.

## 2. Errors and Monitoring
**Goal:** watch jobs + resources, read failures, recover.
- `slurm/03_monitor.md` — `squeue`, `scontrol`, `scancel`; live GPU with
  `nvidia-smi` after SSH-ing to the node; `sinfo`; post-run metrics with `sacct`.
- Logs vs. errors: `02_finetune.sbatch` splits `--output`/`--error`. Demo a CUDA
  OOM by raising `--batch-size`, read the `.err`, fix it.
- `slurm/06_checkpoint.sbatch` — recovering from preemption (intro to
  checkpointing; full treatment in session 5).

## 3. Data Management
**Goal:** move/organize/store data; scratch vs. long-term; node-local.
- `config/paths.sh` — `INPUT_DIR` (read-only project data) vs. `WORK_DIR` (your
  scratch); the "don't write to home" rule.
- Getting data on/off the cluster: `scp` / `sftp` / `rsync` the `input/` tree.
- `src/build_manifest.py` — a manifest decouples *where* data lives from *what* a
  job reads; the natural place to stage to **node-local** storage for the
  I/O-heavy 1,000-image read, then write masks back to project storage.

## 4. Building Workflows
**Goal:** parallelize with job arrays + dependencies.
- Batch vs. interactive recap (`02_finetune.sbatch` vs. `01_interactive.md`).
- `slurm/04_infer_array.sbatch` — segment 1,000 images as a **job array**
  (`--array=0-9`); `seg_infer.py` strides the manifest by `SLURM_ARRAY_TASK_ID`.
- `slurm/05_morphology_dep.sbatch` — the analysis stage runs via
  `--dependency=afterok:` only after every array task succeeds. This is the full
  segment → measure → classify → evaluate pipeline (classify is supervised on the
  survey quality label, grouped by organoid; `evaluate.py` writes the confusion
  matrix + feature-importance plots and reports balanced accuracy / TNR — a good
  moment to show why raw accuracy misleads on this imbalanced QC task).

## 5. Reproducibility and Checkpointing
**Goal:** pin environments, manifests, record metadata, checkpoint.
- `environment.yml` — pinned (`==`) deps. **War story:** the production
  segmenter needs a totally different, frozen env (`mmcv_env`: Python 3.9,
  torch 1.10, `mmcv==2.0.0rc4`) — show why pinning is non-negotiable.
- `src/build_manifest.py` output — the **manifest file** records exactly which
  inputs a run consumed (ties to the AI-Science reproducibility guide).
- **Label provenance** — `add_survey_labels.py` keeps `survey_votes` /
  `survey_n_evaluations` / `survey_label_source` ("direct" vs "propagated")
  alongside each label, so the majority-vote derivation is auditable, not a
  black box. A good reproducibility talking point: labels are *derived*, and the
  derivation is recorded.
- Record job metadata into results: stamp `SLURM_JOB_ID` / git SHA / env hash
  into the run dir (extend `seg_finetune.py`'s save).
- `src/seg_finetune.py --resume` + `slurm/06_checkpoint.sbatch` — checkpoint
  every epoch, `--requeue` + `--signal=B:USR1@120`, resume after preemption.

## Suggested live demo order
```bash
# S1: interactive (01_interactive.md), then:
sbatch slurm/02_finetune.sbatch
# S2: squeue / sacct / nvidia-smi while it runs (03_monitor.md)
# S4: the dependency chain
fid=$(sbatch --parsable slurm/02_finetune.sbatch)
aid=$(sbatch --parsable --dependency=afterok:$fid slurm/04_infer_array.sbatch)
sbatch --dependency=afterok:$aid slurm/05_morphology_dep.sbatch
# S5: cancel+requeue 06_checkpoint.sbatch, watch it resume
```

## Order of execution

```bash
0. setup env          (once)
1. build_manifest     train manifest          ─┐
2. seg_finetune       → seg_finetune_model_final.pt │ GPU
3. build_manifest     infer manifest           │
4. seg_infer          → masks/                 ─┘ GPU (parallel)
5. morphology         → morphology.csv         ─┐
6. classify (+labels) → classified.csv          │ CPU
7. evaluate           → eval/ (metrics + plots) ─┘
   (labels.csv ships with the dataset: add_survey_labels.py → make_labels.py)
```

# Workshop Guide — mapping the project to the 5 lessons

One project, used end to end. Each lesson adds one layer of cluster skill on top
of the same learned-consensus river discharge task.

## 1. Getting Started

**Goal:** Log in to the cluster, set up a software environment, and submit your
first interactive and batch jobs.

- **`slurm/01_interactive.md`** - Log in (SSH + Duo on RCC), clone the repo, build the env, grab a GPU with `srun --pty`, and train on one sub-basin as a smoke test.
- **`environment.yml`** - Modules vs. conda environments; the pinned spec every job activates.
- **`config/setup_env.sh`** - Builds (or rebuilds) the conda env in scratch space; idempotent, and picks whichever package manager is available.
- **`slurm/02_train.sbatch`** - Turn the interactive run into a submitted batch job: accounts, partitions, `--gres=gpu:1`, `sbatch`.

## 2. Data Management

**Goal:** Move data on and off the cluster and place it across scratch,
long-term, and node-local storage for fast, safe runs.

- **`config/paths.sh`** - `INPUT_DIR` (read-only project inputs) vs. `WORK_DIR` (your scratch); the "don't write to `$HOME`" rule, sourced by every job.
- **Getting data on/off the cluster** - `scp` / `sftp` / `rsync` the `input/` tree: `eu_SOS_mini.nc`, `svs_mini.nc`, `priors_mini.csv`.
- **`slurm/06_nodelocal_predict.sbatch`** - Stage inputs onto node-local disk, write the many small per-shard prediction files there, aggregate locally, then copy only `discharge.csv` back to durable storage.
- **`src/build_manifest.py`** - A manifest decouples *where* data lives from *what* a job reads, which is what makes node-local staging and array sharding possible.
- **`config/experiments/baseline.yaml`** + **`src/config.py`** - Keep the config with the results: `save_config()` writes the fully-resolved knobs to `config_used.yaml` in the run's output directory.

## 3. Errors and Monitoring

**Goal:** Watch jobs and GPU usage while they run, read metrics and logs to
diagnose failures, and recover from interruptions.

- **`slurm/03_monitor.md`** - `squeue`, `scontrol`, `scancel`, `sinfo`; live GPU with `nvidia-smi` via `srun --jobid --overlap`; post-run metrics with `sacct`.
- **Logs vs. errors** - `02_train.sbatch` splits `--output` and `--error`. Demo an out-of-memory failure by shrinking `--mem` (or raising the batch size), read the `.err`, fix it.
- **`slurm/03_checkpoint.sbatch`** - Recovering from preemption and the `--time` wall; send the warning yourself with `scancel -s USR2 -b <JOBID>` and watch the job requeue (intro to checkpointing; full treatment in Lesson 5).

## 4. Building Workflows

**Goal:** Turn a multi-step pipeline into parallel job arrays chained by SLURM
dependencies.

- **Batch vs. interactive recap** - `02_train.sbatch` vs. `01_interactive.md`.
- **`slurm/04_predict_array.sbatch`** - Predict discharge for every reach as a **job array** (`--array=0-9`); `src/predict_discharge.py` strides the manifest by `SLURM_ARRAY_TASK_ID` so each task takes its own slice.
- **`slurm/05_aggregate_dep.sbatch`** - The analysis tail runs via `--dependency=afterok:` only after every array task succeeds: `aggregate_discharge.py` gathers the per-shard CSVs, `benchmark_consensus.py` scores the learned consensus against the naive multi-algorithm mean and Confluence's own consensus (NSE / KGE / RMSE / %bias), and `evaluate.py` writes the skill bars, predicted-vs-observed scatter, hydrograph, and `summary.txt` — a good moment to show why one metric alone can mislead.

## 5. Reproducibility and Checkpointing

**Goal:** Pin environments, record manifests and job metadata, and checkpoint
jobs so runs are repeatable and interruption-proof.

- **`environment.yml`** - Pinned deps, so the same spec rebuilds the same environment months later.
- **`config/experiments/baseline.yaml`** - Every model and training knob lives in a committed YAML, so a new experiment is a new committed file that Git diffs cleanly — not an undocumented CLI flag.
- **`src/config.py`** - `save_config()` records the fully-resolved config next to the results, so each run is self-describing.
- **`src/build_manifest.py`** output - The manifest file records exactly which inputs a run consumed (ties to the AI-Science reproducibility guide).
- **Job metadata in results** - Extension exercise: stamp `SLURM_JOB_ID`, the git SHA, and an env hash into the run directory alongside `config_used.yaml` (extend `train_consensus.py`'s save).
- **`src/train_consensus.py --resume-from` + `slurm/03_checkpoint.sbatch`** - Checkpoint on a step interval, at each epoch boundary, and on the `SIGUSR2` warning; `--requeue` + `--signal=B:USR2@120` plus `scontrol requeue` then resume from `consensus_last.pt`.

## Suggested live demo order

```bash
# L1: interactive walkthrough (01_interactive.md), then a first batch job:
sbatch slurm/02_train.sbatch
# L2: stage to node-local disk and copy only the result back
sbatch slurm/06_nodelocal_predict.sbatch
# L3: squeue / sacct / nvidia-smi while it runs (03_monitor.md)
# L4: the dependency chain
fid=$(sbatch --parsable slurm/02_train.sbatch)
aid=$(sbatch --parsable --dependency=afterok:$fid slurm/04_predict_array.sbatch)
sbatch --dependency=afterok:$aid slurm/05_aggregate_dep.sbatch
# L5: signal + requeue 03_checkpoint.sbatch, watch it resume
sbatch slurm/03_checkpoint.sbatch
scancel --signal=USR2 --batch <JOBID>
```

`02_train.sbatch` and `03_checkpoint.sbatch` are **alternatives** for the training
stage — both produce the same `consensus_model_final.pt`.

## Order of execution

```bash
0. setup_env             (once)
1. build_manifest        train manifest           ─┐
2. train_consensus       → consensus_model_final.pt │ GPU
3. build_manifest        infer manifest            │
4. predict_discharge     → predictions_000..009.csv ─┘ GPU (parallel, job array)
5. aggregate_discharge   → discharge.csv           ─┐
6. benchmark_consensus   → scored.csv               │ CPU
7. evaluate              → eval/ (plots + summary) ─┘
```

See `README.md` for the full file table and the copy-pasteable manual commands
for each step.

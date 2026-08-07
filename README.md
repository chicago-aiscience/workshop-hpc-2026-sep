# Workshop HPC — Learned-Consensus River Discharge Reference Project

A small, runnable project used as the through-line for the 5-part HPC workshop
series. It does a real SWOT Confluence task in miniature: **train a small PyTorch
MLP that learns a consensus river discharge from several algorithm estimates,
predict discharge for a batch of river reaches in parallel, then score the
learned consensus against observed gauges** — all on an HPC cluster via SLURM.

This is a *teaching* reimplementation built on **real SWOT Confluence products**
(sliced to a mini two-basin subset). The pipeline consumes the same inputs the
production Confluence pipeline produces — per-reach discharge from four FLPE
algorithms (`metroman`, `momma`, `neobam`, `sic4dvar`), Confluence's own
consensus, and ML-prior flow stats — and learns a gauge-supervised consensus on
top of them. The model itself (a tiny MLP) is a teaching construct chosen to
exercise GPU training, checkpoint/resume, and job arrays; the lesson is about
**running an ML pipeline on the cluster**, not about hydrology or the model.

## The pipeline

```
                            ┌── STAGE A: train (02 or 03) ──────────────────────────┐
  SoS + SVS (+priors) ─► build_manifest.py ─(train)─► train_consensus.py ─► consensus_model_final.pt
                            └───────────────────────────────────────────────────────┘
                                                                                    │
       ┌── STAGE B: predict — job array 0..9 ────────────────────────────────┐      │
       SoS ─► build_manifest.py ─(infer)─► predict_discharge.py (×N shards) ◄─┼──────┘
                                                    │                         │
                                          predictions_000..009.csv            │
       └──────────────────────────────────────────────────────────────────────┘
                                                    │
       ┌── STAGE C: aggregate + score + plot (05, afterok B) ───────────────────────┐
       aggregate_discharge.py ─► discharge.csv ─► benchmark_consensus.py ─► scored.csv
                                       └────────────────► evaluate.py ─► eval/*.png + summary.txt
       └────────────────────────────────────────────────────────────────────────────┘
```

- **Stage A** trains once (one GPU job).
- **Stage B** fans out across a SLURM **job array** — each task predicts a stride
  (`--index`) of the reaches in parallel.
- **Stage C** runs after B via a **job dependency** (`afterok`): it gathers the
  per-shard predictions, scores the learned consensus vs. the naive
  multi-algorithm mean vs. Confluence's consensus, and writes figures.

## Files

| Path | What |
|---|---|
| `environment.yml` | Pinned conda env (`workshop-hpc`) |
| `config/paths.sh` | `INPUT_DIR` / `SOS_FILE` / `SVS_FILE` / `PRIORS_FILE` / `BASINS` / `WORK_DIR` / `ENV_NAME` — sourced by every job |
| `config/setup_env.sh` | Build/rebuild the conda env in scratch space (idempotent) |
| `config/experiments/baseline.yaml` | Committed experiment config: model + training knobs and `data.basins` |
| `src/config.py` | Load/merge config + CLI overrides; write `config_used.yaml` beside each run |
| `src/build_manifest.py` | Join SoS + SVS (+priors) → one row per `(reach, overpass)` manifest CSV |
| `src/_features.py` | Shared feature engineering (log1p, per-row median impute, cyclic month) + target |
| `src/train_consensus.py` | Train the consensus MLP → `consensus_model_final.pt` (checkpoint/resume) |
| `src/predict_discharge.py` | Predict discharge for a `--index` slice of reaches; shardable by job array |
| `src/aggregate_discharge.py` | Concatenate per-shard prediction CSVs → one `discharge.csv` |
| `src/benchmark_consensus.py` | Score learned vs. naive-mean vs. consensus (NSE / KGE / RMSE / %bias) → `scored.csv` |
| `src/evaluate.py` | Figures: skill bars, predicted-vs-observed scatter, hydrograph + `summary.txt` |
| `src/utils.py` | Shared logging helper (`configure_logging`) |
| `slurm/01_interactive.md` | Log in + interactive GPU walkthrough |
| `slurm/02_train.sbatch` | Batch GPU training job (Stage A) |
| `slurm/03_checkpoint.sbatch` | Training with preemption + checkpoint/resume (alt. Stage A) |
| `slurm/03_monitor.md` | `squeue` / `sacct` / `nvidia-smi` cheat-sheet |
| `slurm/04_predict_array.sbatch` | Parallel prediction via **job array** (Stage B) |
| `slurm/05_aggregate_dep.sbatch` | Aggregate → benchmark → evaluate via **job dependency** (Stage C) |
| `slurm/06_nodelocal_predict.sbatch` | Predict + aggregate on **node-local** disk, copy only the result back |
| `WORKSHOP_GUIDE.md` | Which file teaches which lesson |

## Quick start (on the cluster)

```bash
conda env create -f environment.yml && conda activate workshop-hpc
mkdir -p logs
# Train, then run the parallel prediction → analysis chain:
fid=$(sbatch --parsable slurm/02_train.sbatch)
aid=$(sbatch --parsable --dependency=afterok:$fid slurm/04_predict_array.sbatch)
sbatch --dependency=afterok:$aid slurm/05_aggregate_dep.sbatch
```

`02_train.sbatch` and `03_checkpoint.sbatch` are **alternatives** — use `03` on a
preemptible partition to get SIGUSR2 checkpoint-on-preemption + automatic
requeue; both produce the same `consensus_model_final.pt`.

If scratch is periodically purged, build the env there instead with
`SCRATCH_BASE=/path/to/scratch ./config/setup_env.sh` (idempotent; see the
script header for options).

## Data

Defaults come from `config/paths.sh` (`INPUT_DIR=/data/workshop-hpc-data/confluence-mini/input`),
the mini dataset produced once by a script from the full public SWOT Confluence products, sliced to two basins:

```
input/eu_SOS_mini.nc     # SoS results: per-reach FLPE discharge (metroman, momma,
                         #   neobam, sic4dvar) + Confluence consensus_q
input/svs_mini.nc        # SWOT Validation Set: observed daily gauge discharge,
                         #   keyed by reach_id (v16) -- the supervised TARGET
input/priors_mini.csv    # per-reach ML-prior flow stats (mean_q, monthly_q); optional
```

Everything **joins on `reach_id` (v16)**. The SoS gives per-reach overpass *times*
(seconds since 2000-01-01); the SVS gives *daily* gauge discharge; `build_manifest.py`
matches each overpass to the gauge value on the same calendar day.

- `--mode train` keeps only gauged rows with **≥2 valid algorithm** estimates
  (real supervised targets).
- `--mode infer` keeps **all reaches** in the basins (no target filtering), for
  the prediction pass.

`BASINS="2322 2326"` are SWORD Pfafstetter prefixes (2322 = Loire, 2326 = Rhine);
a reach is included if its id starts with any listed prefix. **Predicted discharge
is an output** (`predicted_q`, written by `predict_discharge.py`), never an input.

## Manual step-by-step execution

```bash
cd workshop-hpc

# 0. Environment (once)
conda env create -f environment.yml
conda activate workshop-hpc
source config/paths.sh          # sets INPUT_DIR, SOS_FILE, SVS_FILE, PRIORS_FILE, BASINS, WORK_DIR

# 1. Training manifest: one row per (reach, overpass), gauged rows with >=2 algorithms
python src/build_manifest.py --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --basins $BASINS --mode train --out "$WORK_DIR/data/manifest_train.csv"

# 2. Train the consensus MLP (GPU) -> $WORK_DIR/runs/consensus/consensus_model_final.pt
python src/train_consensus.py --manifest "$WORK_DIR/data/manifest_train.csv" \
    --out "$WORK_DIR/runs/consensus" --epochs 500 --batch-size 256

# 3. Inference manifest: every reach in the basins (no target filtering)
python src/build_manifest.py --sos "$SOS_FILE" --svs "$SVS_FILE" --priors "$PRIORS_FILE" \
    --basins $BASINS --mode infer --out "$WORK_DIR/data/manifest_infer.csv"

# 4. Predict discharge (GPU) -> $WORK_DIR/runs/predictions/predictions_000.csv
#    (in SLURM this fans out across a job array; --index/--num-tasks stride the manifest)
python src/predict_discharge.py --manifest "$WORK_DIR/data/manifest_infer.csv" \
    --model "$WORK_DIR/runs/consensus/consensus_model_final.pt" \
    --out-dir "$WORK_DIR/runs/predictions" --index 0 --num-tasks 1

# 5. Aggregate per-shard predictions -> one tidy discharge table
python src/aggregate_discharge.py --pred-dir "$WORK_DIR/runs/predictions" \
    --out "$WORK_DIR/runs/discharge.csv"

# 6. Score learned vs. naive-mean vs. consensus against the gauge -> scored.csv
python src/benchmark_consensus.py --discharge "$WORK_DIR/runs/discharge.csv" \
    --out "$WORK_DIR/runs/scored.csv"

# 7. Evaluate + plot -> skill bars, pred-vs-obs scatter, hydrograph, summary.txt
python src/evaluate.py --discharge "$WORK_DIR/runs/discharge.csv" \
    --scored "$WORK_DIR/runs/scored.csv" --out-dir "$WORK_DIR/runs/eval"
```

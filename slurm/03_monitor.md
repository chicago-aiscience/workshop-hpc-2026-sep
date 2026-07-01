# Workshop 2 — Monitoring, metrics, and recovery

Use the fine-tune job (`02_finetune.sbatch`) as the running example.

## Submit and watch the queue

```bash
sbatch slurm/02_finetune.sbatch       # prints: Submitted batch job 12345
squeue --me                           # your jobs: state (PD=pending, R=running)
squeue -j 12345 --long                # one job, detailed
scontrol show job 12345               # full record: node, resources, reason
```

## Watch resources live (GPU usage)

Find the node your job landed on (`squeue --me` → NODELIST), then SSH to it and
watch the GPU:

```bash
squeue --me -o "%N"                   # e.g. gpu-a001
ssh gpu-a001
nvidia-smi -l 1                       # refresh every 1s: util %, memory, the python PID
```

Cluster-wide view of what's free:

```bash
sinfo -p general                      # partition state / idle nodes
sinfo -N -o "%N %G %C %m"             # per-node: gres(GPU), CPUs, memory
```

## Logs vs. errors

The job splits output by stream (see the `#SBATCH --output/--error` lines):

```bash
tail -f logs/finetune_12345.out       # stdout: loss curve, progress
tail -f logs/finetune_12345.err       # stderr: tracebacks, CUDA OOM, warnings
```

Debugging rule of thumb: **`.out` tells you how far it got, `.err` tells you why
it stopped.** A CUDA out-of-memory error here → lower `--batch-size`.

## Metrics after the job finishes

```bash
sacct -j 12345 --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES%40
```

`MaxRSS` (peak memory) and `Elapsed` tell you whether your `--mem`/`--time`
requests were right-sized for next time.

## Cancel

```bash
scancel 12345                         # one job
scancel --me                          # all of yours
```

## Recovering from interruption / preemption

The DSI cluster can **preempt** your job (see
<https://cluster-policy.ds.uchicago.edu/using-the-cluster/batch-jobs/>). Our
training script checkpoints every epoch and `--resume` reloads it, so a
requeued job continues instead of restarting. See `06_checkpoint.sbatch`.

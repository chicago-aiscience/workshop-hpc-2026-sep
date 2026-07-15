# Workshop 2 — Monitoring, metrics, and recovery

Use the fine-tune job (`02_finetune.sbatch`) as the running example.

## Submit and watch the queue

```bash
sbatch slurm/02_finetune.sbatch       # prints: Submitted batch job 12345
squeue --me                           # your jobs, incl. the ST (state) column
squeue -j 12345 --long                # one job, detailed
scontrol show job 12345               # full record: node, resources, reason
```

## Job state codes

The `ST` column in `squeue` gives the state of your job. The codes you'll see most:

| Code | State      | Meaning                                                                                     |
| ---- | ---------- | ------------------------------------------------------------------------------------------- |
| `PD` | Pending    | Awaiting resource allocation. Check `NodeList(Reason)` to see why it hasn't started.        |
| `R`  | Running    | Started and holding the resources requested in the `#SBATCH` directives.                    |
| `CG` | Completing | Job is finishing (cleaning up).                                                             |
| `CD` | Completed  | Finished successfully (exit code 0).                                                        |
| `F`  | Failed     | Terminated with a non-zero exit code or other failure condition.                            |
| `CA` | Cancelled  | Cancelled by you (`scancel`) or an admin, before or during the run.                         |
| `OOM`| Out of Memory | Killed after hitting an out-of-memory error — raise `--mem` or lower `--batch-size`.     |
| `TO` | Timeout    | Terminated after reaching its `--time` limit.                                               |

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

## Debugging

### Why is my job stuck in `PD` (pending)?

A pending job isn't broken — SLURM just hasn't scheduled it yet. The
`NodeList(Reason)` column tells you why:

```bash
squeue --me -o "%.10i %.8T %.40R"     # jobid, state, and the reason
scontrol show job 12345 | grep -i reason
```

[Reasons list](https://slurm.schedmd.com/squeue.html#SECTION_JOB-REASON-CODES)

### What was the exit code?

For a job that ran (or failed), the exit code tells you *how* it ended:

```bash
sacct -j 12345 --format=JobID,State,ExitCode,DerivedExitCode,Elapsed
scontrol show job 12345 | grep -i exit    # only for ~5 min after it ends
```

## Cancel

```bash
scancel 12345                         # one job
scancel --me                          # all of yours
```

## Recovering from interruption / preemption

The DSI cluster can **preempt** your job (see
<https://cluster-policy.ds.uchicago.edu/using-the-cluster/batch-jobs/>). Our
training script checkpoints on a regular interval, at each epoch boundary, and
on the preemption warning, then hands the checkpoint back with `--resume-from`
on restart — so a requeued job continues instead of restarting. See
`03_checkpoint.sbatch`.

### Test it yourself — send the warning signal

You don't have to wait for a real preemption (or the `--time` wall) to see this
work. The `#SBATCH --signal=B:USR2@120` line only delivers `SIGUSR2` automatically
120s before the time limit, but you can send that same signal yourself at any time
with `scancel`:

```bash
scancel --signal=USR2 --batch 12345   # short form: scancel -s USR2 -b 12345
```

> 🧰 - The `--batch` (`-b`) flag is essential. Without it, `scancel --signal` targets
> the job's `srun` steps — and this job launches Python directly (no `srun` step), so
> the signal would land on nothing. `--batch` sends it to the batch script, where the
> `USR2` trap lives (the runtime counterpart of the `B:` in `--signal=B:USR2`).

Full loop — the wrapper requeues itself, so resume is automatic (same job id):

```bash
# 1. Submit and note the job id
sbatch slurm/03_checkpoint.sbatch          # -> Submitted batch job 12345

# 2. Wait until it is RUNNING and a few steps in (so there is progress to save)
squeue --me
tail -f logs/ckpt_12345.out

# 3. Send the preemption warning yourself
scancel --signal=USR2 --batch 12345

# 4. In logs/ckpt_12345.out you should see, in order:
#      USR2 received: signalling Python to checkpoint...
#      [seg] signal 12 received; checkpointing at next step boundary...
#      [seg] checkpoint saved on preemption warning; exiting for requeue.
#      Requeuing job 12345 via scontrol requeue.

# 5. Watch it come back on its own: squeue shows the SAME job id go PD -> R again,
#    and the new run appends "Resuming from .../seg_finetune_last.pt" to the log.
squeue --me
sacct -j 12345 --format=JobID,State,ExitCode,Restart%7   # Restart increments to 1
ls -l "$WORK_DIR/runs/seg/seg_finetune_last.pt"
```

Notes:
- `SIGUSR2` is **signal 12** on Linux (what you'll see logged). The job must be `R`
  (running), not `PD`, when you signal it. We use USR2 (not USR1) because some
  frameworks (PyTorch Lightning, submitit) already claim USR1 for their own requeue.
- **The requeue uses `scontrol requeue`, which works on any cluster** (given
  `#SBATCH --requeue` and that you own the job) — no special exit-code policy needed.
  If requeue is disabled on your cluster, you can still resume manually: `scancel
  12345`, then `sbatch slurm/03_checkpoint.sbatch` (the wrapper reads the checkpoint
  off disk).
- To signal the process directly instead: `squeue --me -o "%N"` → `ssh <node>` →
  `kill -USR2 <python_pid>`. `scancel -b` is cleaner and needs no node access.

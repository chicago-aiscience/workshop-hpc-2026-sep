#!/usr/bin/env python3
"""Train the learned-consensus discharge model.

Fits a small PyTorch MLP that predicts observed gauge discharge from the FLPE
algorithm estimates (metroman, momma, neobam, sic4dvar), Confluence's consensus,
and ML-prior flow stats -- a *learned* multi-model consensus we later show beating
the naive multi-algorithm mean. Reads the (reach, overpass) rows from a manifest
built by build_manifest.py; both features and target are log-transformed.

Supports checkpoint/resume so a preempted job can pick up where it left off
(Workshop 2 + 5): the SIGUSR2 warning triggers a safe save + clean exit, and the
wrapper in slurm/03_checkpoint.sbatch requeues.

    python src/train_consensus.py --manifest data/manifest_train.csv \
        --out runs/consensus --epochs 50
"""
import argparse
import datetime
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import configure_logging

# _features lives beside this script; import works when run as `python src/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _features import make_features, make_target  # noqa: E402

CHECKPOINT_INTERVAL: int = 50

logger = configure_logging("train_consensus")

# Set by the SIGUSR2 handler when SLURM warns of imminent preemption / time-out.
# (SIGUSR2 not SIGUSR1: SIGUSR1 is already claimed by some frameworks.)
should_checkpoint: bool = False


def handle_preempt(signum, frame) -> None:
    """Signal handler for the preemption warning (SIGUSR2): flip a flag only.

    Saving here would be unsafe -- a signal can interrupt mid-instruction and
    torch.save is not async-signal-safe -- so the training loop saves at the next
    step boundary, where state is consistent.
    """
    global should_checkpoint
    logger.info(f"[train] signal {signum} received; checkpointing at next step boundary...")
    should_checkpoint = True


def atomic_torch_save(state: dict, path: Path) -> None:
    """Write a checkpoint atomically (temp file + os.replace) so a kill can't corrupt it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


class ConsensusMLP(nn.Module):
    """Small MLP with built-in input standardization.

    The per-feature mean/std are stored as buffers, so they travel inside the
    saved state_dict -- prediction loads the model and gets the exact same
    normalization without a separate scaler file.
    """

    def __init__(self, n_features: int, mean: np.ndarray, std: np.ndarray, hidden: int = 64) -> None:
        """Build the network and register the input-normalization buffers.

        Args:
            n_features: Number of input features.
            mean: Per-feature means computed on the training set.
            std: Per-feature standard deviations (zeros replaced with 1).
            hidden: Width of the two hidden layers.
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(np.where(std > 0, std, 1.0), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize inputs, then predict log1p(discharge) as a length-n vector."""
        return self.net((x - self.mean) / self.std).squeeze(-1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a training run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/consensus")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-dir", default=None,
                    help="directory for the rolling consensus_last.pt (defaults to --out)")
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint path to resume from; ignored if it does not exist")
    ap.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL,
                    help="save a regular checkpoint every N steps (0 disables interval saves)")
    return ap.parse_args()


def load_checkpoint(ckpt: Path, device: str, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    """Restore model + optimizer state in place; return the epoch to resume from."""
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    epoch = state["epoch"]
    start_epoch = epoch + 1 if state.get("epoch_completed", True) else epoch
    logger.info(f"[train] resumed from epoch {start_epoch}")
    return start_epoch


def train_model(start_epoch: int, num_epochs: int, model: nn.Module, loader: DataLoader,
                device: str, opt: torch.optim.Optimizer, loss_fn: nn.Module, ckpt: Path,
                names: list[str], checkpoint_interval: int) -> None:
    """Run the training loop with interval, preemption, and epoch checkpoints.

    A checkpoint is written to the rolling `ckpt` path (1) on the SIGUSR2
    preemption warning -- then exit cleanly for the wrapper to requeue, (2) every
    `checkpoint_interval` steps, and (3) at each epoch end (marked
    `epoch_completed`). Feature names are stored so prediction reads the right
    columns.

    Args:
        start_epoch: First epoch index to run.
        num_epochs: Exclusive upper bound on epoch index.
        model: The MLP, already on `device`.
        loader: DataLoader of `(features, log_target)` batches.
        device: "cuda" or "cpu".
        opt: Optimizer.
        loss_fn: Regression loss on the log target.
        ckpt: Rolling checkpoint path.
        names: Feature column names (persisted for prediction).
        checkpoint_interval: Save every N steps (0 disables interval saves).
    """
    def snapshot(epoch: int, step: int, done: bool) -> dict:
        return {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                "step": step, "epoch_completed": done, "feature_names": names}

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running = 0.0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()

            if should_checkpoint:                       # 1. preemption warning
                atomic_torch_save(snapshot(epoch, step, False), ckpt)
                logger.info("[train] checkpoint saved on preemption warning; exiting for requeue.")
                sys.exit(0)
            if checkpoint_interval and step > 0 and step % checkpoint_interval == 0:
                atomic_torch_save(snapshot(epoch, step, False), ckpt)   # 2. interval

        logger.info(f"[train] epoch {epoch}  loss={running / len(loader):.4f}")
        atomic_torch_save(snapshot(epoch, len(loader) - 1, True), ckpt)  # 3. epoch end


def build_dataloader(manifest: str, batch_size: int
                     ) -> tuple[DataLoader, list[str], np.ndarray, np.ndarray]:
    """Read the manifest and build the training loader + standardization stats.

    Args:
        manifest: Path to the training manifest CSV from `build_manifest.py`.
        batch_size: Rows per training batch.

    Returns:
        `(loader, names, mean, std)`: a shuffled DataLoader of
        `(features, log_target)` batches, the feature column names, and the
        per-feature mean/std used to standardize inputs inside the model.
    """
    df = pd.read_csv(manifest)
    X, names = make_features(df)   # model inputs (log-transformed features)
    y = make_target(df)            # ground-truth log1p gauge discharge
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    mean, std = X.mean(axis=0), X.std(axis=0)
    logger.info(f"[train] rows={len(df)}  features={len(names)}")
    return loader, names, mean, std


def build_model(n_features: int, mean: np.ndarray, std: np.ndarray, lr: float, device: str
                ) -> tuple[nn.Module, torch.optim.Optimizer, nn.Module]:
    """Build the MLP (with standardization buffers), Adam optimizer, and MSE loss."""
    model = ConsensusMLP(n_features, mean, std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    return model, opt, loss_fn


def prepare_checkpointing(args: argparse.Namespace, out: Path, device: str,
                          model: nn.Module, opt: torch.optim.Optimizer) -> tuple[Path, int]:
    """Resolve the rolling checkpoint path and the epoch to resume from.

    Restores model/optimizer state in place when `--resume-from` points at an
    existing checkpoint; otherwise starts fresh at epoch 0.

    Returns:
        `(ckpt, start_epoch)`: the rolling `consensus_last.pt` path and the first
        epoch index to run.
    """
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else out
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "consensus_last.pt"

    start_epoch = 0
    if args.resume_from and Path(args.resume_from).exists():
        start_epoch = load_checkpoint(Path(args.resume_from), device, model, opt)
    return ckpt, start_epoch


def save_final_model(model: nn.Module, names: list[str], out: Path) -> Path:
    """Save the final weights + feature names (so prediction reads the right columns)."""
    path = out / "consensus_model_final.pt"
    torch.save({"model": model.state_dict(), "feature_names": names,
                "target": "log1p_gauge_q"}, path)
    return path


def main() -> None:
    """Load the manifest, build features/model, optionally resume, then train."""
    start = datetime.datetime.now()
    args = parse_args()
    for name, value in vars(args).items(): logger.info("%s = %s", name, value)

    signal.signal(signal.SIGUSR2, handle_preempt)   # checkpoint on preemption warning

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[train] device={device}")

    loader, names, mean, std = build_dataloader(args.manifest, args.batch_size)
    model, opt, loss_fn = build_model(len(names), mean, std, args.lr, device)
    ckpt, start_epoch = prepare_checkpointing(args, out, device, model, opt)

    train_model(start_epoch, args.epochs, model, loader, device, opt, loss_fn,
                ckpt, names, args.checkpoint_interval)

    final = save_final_model(model, names, out)
    logger.info(f"[train] done -> {final}  ({datetime.datetime.now() - start})")


if __name__ == "__main__":
    main()

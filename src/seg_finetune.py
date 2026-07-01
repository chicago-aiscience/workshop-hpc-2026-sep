#!/usr/bin/env python3
"""Fine-tune a pretrained segmentation model on organoid images.

Teaching version of the production `pipeline/images/segmentation_mmseg/` code:
same idea (take an ImageNet-pretrained backbone, fine-tune it to a 2-class
organoid / background segmenter) but written as one readable PyTorch loop
instead of an MMSegmentation config. Uses torchvision's FCN-ResNet50, which is
already in `core_env` -- no new libraries.

Reads (image_path, mask_path) pairs from a manifest CSV built by
build_manifest.py. Supports checkpoint/resume so a preempted job can pick up
where it left off (Workshop 2 + 5).

    python src/seg_finetune.py --manifest data/manifest_train.csv \
        --out runs/seg --epochs 10
"""
import argparse
import csv
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_tensor, resize, normalize

# Per-channel (R, G, B) statistics the pretrained backbone was trained on; the
# input must be normalized with these for the pretrained weights to behave.
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]


class OrganoidSeg(Dataset):
    """Image/mask pairs from a manifest CSV (columns: image_path, mask_path).

    Each item is one training example: a normalized RGB image tensor and its
    integer per-pixel label map (0 = background, 1 = organoid).
    """

    def __init__(self, manifest: str, size: tuple[int, int] = (384, 512)) -> None:
        """Load the manifest rows into memory.

        Args:
            manifest: Path to the CSV with `image_path` and `mask_path` columns.
            size: Target (height, width) every image and mask is resized to so
                samples can be batched together.
        """
        self.rows: list[dict[str, str]] = list(csv.DictReader(open(manifest)))
        self.size = size

    def __len__(self) -> int:
        """Return the number of (image, mask) pairs in the dataset."""
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, and tensorize the i-th image/mask pair.

        Args:
            i: Row index into the manifest.

        Returns:
            A tuple `(x, y)` where `x` is the normalized image tensor of shape
            `[3, H, W]` (float) and `y` is the label map of shape `[H, W]` (int64,
            values in {0, 1}). The image is bilinearly resized then normalized;
            the mask is nearest-neighbor resized then binarized so labels stay
            crisp and aligned with the image.
        """
        row = self.rows[i]
        img = Image.open(row["image_path"]).convert("RGB")
        mask = Image.open(row["mask_path"]).convert("L")
        x = resize(to_tensor(img), self.size)
        x = normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        m = resize(Image.fromarray(np.array(mask)), self.size, interpolation=0)
        y = torch.from_numpy((np.array(m) > 0).astype("int64"))  # binarize -> {0,1}
        return x, y


def parse_commandline_args() -> argparse.Namespace:
    """Parse command-line arguments for a fine-tuning run.

    Returns:
        The parsed arguments: `manifest`, `out`, `epochs`, `batch_size`, `lr`,
        and the `resume` flag.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/seg")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true", help="resume from out/last.pt if present")
    args = ap.parse_args()
    return args


def build_model() -> nn.Module:
    """Build a pretrained FCN-ResNet50 with the head swapped for 2 classes.

    Loads the COCO/ImageNet-pretrained weights, then replaces the final 1x1 conv
    of both the main and auxiliary FCN heads so the model outputs 2 classes
    (background, organoid) instead of the pretrained 21.

    Returns:
        The segmentation model, ready to move to a device and fine-tune.
    """
    model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    return model


def load_checkpoint(
    start_epoch: int,
    ckpt: Path,
    device: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state from a checkpoint, in place.

    Args:
        start_epoch: Ignored; the resume epoch is read from the checkpoint.
        ckpt: Path to the `.pt` checkpoint saved by `train_model`.
        device: Device string ("cuda" or "cpu") to map the tensors onto.
        model: Model whose weights are loaded in place.
        opt: Optimizer whose state is loaded in place.

    Returns:
        The epoch to resume training from (checkpoint epoch + 1).
    """
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    start_epoch = state["epoch"] + 1
    print(f"[seg] resumed from epoch {start_epoch}")
    return start_epoch


def train_model(
    start_epoch: int,
    num_epochs: int,
    model: nn.Module,
    loader: DataLoader,
    device: str,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    ckpt: Path,
) -> None:
    """Run the fine-tuning loop, checkpointing after every epoch.

    For each epoch, runs the predict -> loss -> backprop -> step cycle over every
    batch, prints the mean epoch loss, and saves a checkpoint so a preempted job
    can resume via `load_checkpoint`.

    Args:
        start_epoch: First epoch index to run (0, or the resume point).
        num_epochs: Stop before this epoch index (exclusive upper bound).
        model: The segmentation model to train, already on `device`.
        loader: DataLoader yielding `(image, label)` batches.
        device: Device string ("cuda" or "cpu") batches are moved to.
        opt: Optimizer applied each step.
        loss_fn: Per-pixel loss comparing logits to the label map.
        ckpt: Path the per-epoch checkpoint is written to.
    """
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()               # clear last steps gradients
            out_logits = model(x)["out"]  # forward: predict masks
            loss = loss_fn(out_logits, y) # measure error vs. ground truth
            loss.backward()               # backward: compute gradient for every weight
            opt.step()                    # optimizer: nudge every weight to reduce loss
            running += loss.item()        # track loss
        avg = running / len(loader)
        print(f"[seg] epoch {epoch}  loss={avg:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, ckpt)  # checkpoint every epoch


def main() -> None:
    """Entry point: set up data/model/optimizer, optionally resume, then train.

    Builds the dataloader, model, and optimizer from the parsed arguments;
    resumes from `out/last.pt` if `--resume` is set and a checkpoint exists;
    runs training; then saves the final weights to `out/seg_finetune_model_final.pt`.
    """
    start = datetime.datetime.now()

    # Define command line arguments
    args = parse_commandline_args()

    # Set up output directory and detect GPU vs. CPU device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[seg] device={device}  manifest={args.manifest}")

    # Load the input data and define the model for training
    loader = DataLoader(OrganoidSeg(args.manifest), batch_size=args.batch_size,
                        shuffle=True, num_workers=4)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Determine if an existing run was interrupted and load checkpoint
    start_epoch = 0
    ckpt = out / "seg_finetune_last.pt"
    if args.resume and ckpt.exists():  # checkpoint recovery (preemption)
        start_epoch = load_checkpoint(start_epoch, ckpt, device, model, opt)

    # Train the model
    train_model(start_epoch, args.epochs, model, loader, device, opt, loss_fn, ckpt)

    # Save the final model state
    torch.save(model.state_dict(), out / "seg_finetune_model_final.pt")
    print(f"[seg] done -> {out/'seg_finetune_model_final.pt'}")

    end = datetime.datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()

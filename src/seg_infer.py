#!/usr/bin/env python3
"""Run the fine-tuned segmenter over a list of images -> write predicted masks.

Designed to be sharded by a SLURM job array: pass --shard / --num-shards and
each array task segments its own slice of the manifest (Workshop 4).

    python src/seg_infer.py --manifest data/manifest_infer.csv \
        --model runs/seg/seg_finetune_model_final.pt --out-dir runs/masks \
        --shard ${SLURM_ARRAY_TASK_ID:-0} --num-shards ${SLURM_ARRAY_TASK_COUNT:-1}
"""
import argparse
import csv
import datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize, normalize

from seg_finetune import build_model, IMAGENET_MEAN, IMAGENET_STD


def parse_commandline_args() -> argparse.Namespace:
    """Parse command-line arguments for an inference run.

    Returns:
        The parsed arguments: `manifest`, `model`, `out_dir`, and the
        `shard` / `num_shards` job-array slice controls.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="runs/masks")
    ap.add_argument("--shard", type=int, default=0,
                    help="which slice this process handles (0-based)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total number of slices the manifest is split into")
    args = ap.parse_args()
    return args


def main() -> None:
    """Segment this shard's images and write a predicted mask for each.

    Loads the fine-tuned model, takes the `--shard`-th strided slice of the
    manifest (so a SLURM job array can run many of these in parallel), and for
    each image runs a forward pass, converts the per-pixel argmax to a {0, 255}
    mask, resizes it back to the original image size, and saves it as
    `<image-stem>_predmask.png` under `--out-dir`.
    """
    start = datetime.datetime.now()
    # Parse command line arguments
    args = parse_commandline_args()

    # Set up output directory and GPU vs. CPU device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Retrieve list of images and parallel batch to process
    rows = list(csv.DictReader(open(args.manifest)))
    subset = rows[args.shard::args.num_shards]  # strided slice for this array task
    print(f"[infer] shard {args.shard}/{args.num_shards}: {len(subset)} images  device={device}")

    # Build the model, load previous state, and enter evaluation mode
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    with torch.no_grad():  # do not build the autograd graph
        for row in subset:
            img = Image.open(row["image_path"]).convert("RGB")
            w, h = img.size                                                                 # Save original size
            x = normalize(resize(to_tensor(img), (384, 512)), IMAGENET_MEAN, IMAGENET_STD)  # Resize and normalize
            logits = model(x.unsqueeze(0).to(device))["out"]                                # Add a batch dimension and run forward pass
            pred = logits.argmax(1)[0].cpu().numpy().astype("uint8") * 255                  # Pick the winning class and map to black/white mask
            mask = Image.fromarray(pred).resize((w, h), Image.NEAREST)                      # Resize predicted mask back to original size
            name = Path(row["image_path"]).stem + "_predmask.png"
            mask.save(out_dir / name)

    print(f"[infer] shard {args.shard} done -> {out_dir}")

    end = datetime.datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()

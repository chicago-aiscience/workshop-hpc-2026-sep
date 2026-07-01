#!/usr/bin/env python3
"""Build an (image_path[, mask_path]) manifest CSV for training / inference.

A manifest is the reproducibility backbone of the whole project (Workshop 5):
it records *exactly* which files a job consumed, so a run is replayable even if
the data directory changes later.

Both modes read the raw full-resolution images and pair them via
`image_mapping_thresholded_and_manual.json`, which already links each well to
its best-Z raw image (`Best Z Filename`) and its hand-drawn ground-truth mask
(`MT Mask Path`):

  --mode train   emit (image_path, mask_path): raw image + MANUAL mask. This is
                 the real fine-tune -- the model learns from human ground truth.

  --mode infer   emit (image_path): the raw images to segment. No labels; the
                 model produces predicted masks for these.

The json stores cluster-absolute paths (/net/projects2/...). We rebase them onto
$INPUT_DIR by basename so the same manifest builder works locally and on the
cluster.

    python src/build_manifest.py --input-dir "$INPUT_DIR" --mode train \
        --limit 1000 --out data/manifest_train.csv
"""
import argparse
import csv
import datetime
import json
from collections.abc import Iterator
from pathlib import Path

MAPPING = "json/image_mapping_thresholded_and_manual.json"
RAW_SUBDIR = "raw_images"
MASK_SUBDIR = "masks"


def pairs(input_dir: Path, want_mask: bool) -> Iterator[tuple[str, str | None]]:
    """Yield existing (image, mask) pairs from the mapping JSON.

    Reads the mapping JSON under `input_dir`, and for each well rebases the raw
    image (`Best Z Filename`) and manual mask (`MT Mask Path`) onto `input_dir`
    by basename. Entries are skipped if a field is missing, the image file does
    not exist, or (when `want_mask`) the mask file does not exist.

    Args:
        input_dir: Root data directory holding `json/`, `raw_images/`, `masks/`.
        want_mask: If True (train mode), require and return the mask path; if
            False (infer mode), the mask is not required and `None` is returned
            in its place.

    Yields:
        `(image_path, mask_path)` tuples as strings, where `mask_path` is `None`
        when `want_mask` is False.
    """
    mapping = json.load(open(input_dir / MAPPING))
    for rec in mapping.values():
        bz, mt = rec.get("Best Z Filename"), rec.get("MT Mask Path")
        if not bz or not mt:
            continue
        # rebase cluster-absolute paths onto the local/cluster $INPUT_DIR
        img = input_dir / RAW_SUBDIR / Path(bz).name
        mask = input_dir / MASK_SUBDIR / Path(mt).name
        if not img.exists():
            continue
        if want_mask and not mask.exists():
            continue
        yield str(img), (str(mask) if want_mask else None)


def main() -> None:
    """Build a manifest CSV from command-line arguments.

    Parses `--input-dir`, `--out`, `--mode` (train/infer), and `--limit`;
    collects the image (and, in train mode, mask) pairs; optionally caps the row
    count; and writes the CSV with the appropriate header columns.
    """
    start = datetime.datetime.now()

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["train", "infer"], default="train")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (e.g. 1000 for the workshop)")
    args = ap.parse_args()

    want_mask = args.mode == "train"
    rows = list(pairs(args.input_dir, want_mask))
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        if want_mask:
            w.writerow(["image_path", "mask_path"])
            w.writerows(rows)
        else:
            w.writerow(["image_path"])
            w.writerows([(img,) for img, _ in rows])
    print(f"[manifest] {len(rows)} rows ({args.mode}) -> {args.out}")

    end = datetime.datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()

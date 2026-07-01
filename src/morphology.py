#!/usr/bin/env python3
"""Turn predicted masks into per-organoid morphology features.

Mirrors the production `pipeline/images/quality/organoid_area.py` idea (mask ->
cross-sectional area) and adds the circularity measure:

    circularity = 4*pi*area / perimeter**2     (1.0 = perfect circle)

Writes one CSV row per image. This is the input to classify.py and the
dependency job in Workshop 4 (runs after the inference array finishes).

Assumes there is one organoid per well but this may not always be the case.
Simplified for the workshop.

    python src/morphology.py --mask-dir runs/masks --out runs/morphology.csv
"""
import argparse
import csv
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import measure


def features_for_mask(mask: np.ndarray) -> dict[str, float]:
    """Compute shape features for the largest blob in a mask.

    Binarizes the mask, labels its connected components, and measures the
    largest one (assumed to be the organoid). Returns zeros if the mask is
    empty (no foreground detected).

    Args:
        mask: 2D array; any nonzero pixel is treated as foreground.

    Returns:
        A dict with `area_px` (pixel count), `perimeter_px` (boundary length),
        `circularity` (4*pi*area/perimeter**2, capped at 1.0; 1.0 = circle), and
        `eccentricity` (0.0 = circle, ->1.0 = elongated).
    """
    labels = measure.label(mask > 0)      # Connected-component labeling - ids image blobs (organoid)
    properties = measure.regionprops(labels)   # Compute measurements for blobgs

    if not properties:                         # Nothing detected
        return dict(area_px=0, perimeter_px=0.0, circularity=0.0, eccentricity=0.0)

    org_prop = max(properties, key=lambda r: r.area)  # the organoid = biggest blob - may fail on "split" organoids
    perimeter = float(org_prop.perimeter)
    circ = (4 * np.pi * org_prop.area / perimeter**2) if perimeter > 0 else 0.0

    return dict(area_px=int(org_prop.area), perimeter_px=round(perimeter, 2),
                circularity=round(min(circ, 1.0), 4),
                eccentricity=round(float(org_prop.eccentricity), 4))


def main() -> None:
    """Compute morphology features for every predicted mask in a directory.

    Globs `*_predmask.png` under `--mask-dir`, runs `features_for_mask` on each,
    and writes one row per image (keyed by image_id) to the `--out` CSV.
    """
    start = datetime.datetime.now()

    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--out", default="runs/morphology.csv")
    args = ap.parse_args()

    # Retrieve sorted predicted masks
    masks = sorted(Path(args.mask_dir).glob("*_predmask.png"))
    print(f"[morph] {len(masks)} masks from {args.mask_dir}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Calculate morphology data
    with open(args.out, "w", newline="") as f:
        cols = ["image_id", "area_px", "perimeter_px", "circularity", "eccentricity"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in masks:
            arr = np.array(Image.open(m).convert("L"))
            row = {"image_id": m.stem.replace("_predmask", "")}
            row.update(features_for_mask(arr))
            w.writerow(row)

    print(f"[morph] wrote {args.out}")

    end = datetime.datetime.now()
    print(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()

# Workshop HPC — Organoid Segmentation Reference Project

A small, runnable project used as the through-line for the 5-part HPC workshop
series. It does the real Promega task in miniature: **fine-tune a pretrained
segmentation model on organoid microscopy images, segment a batch of images,
then derive morphology (area + circularity) and classify** — all on an HPC
cluster via SLURM.

This is a *teaching* reimplementation of the [production segmentation pipeline](https://github.com/dsi-clinic/2025-promega-mini-test/tree/main/pipeline/images/segmentation_mmseg),
which uses `SegFormer` via `MMSegmentation`. The real Promega segmenter is a
2-class ResNet-50 fine-tuned on hand-drawn organoid masks via MMSegmentation.
The workshop is that exact recipe with the same backbone, same masks, same JSON
pairing but rewritten as ~90 readable lines so the lesson is about running it on
the cluster, not about MMSegmentation.

## The pipeline

```
images ─► build_manifest.py ─► seg_finetune.py ─► seg_infer.py ─► morphology.py ─► classify.py ─► evaluate.py
          (image/mask CSV)     (fine-tune FCN)    (masks)         (area+circ.)    (Acceptable?)   (metrics+plots)
                                                                                   ▲
                                            survey labels (labels.csv) ────────────┘
```

## Files

| Path | What |
|---|---|
| `environment.yml` | Pinned conda env (`workshop-hpc`) |
| `config/paths.sh` | `INPUT_DIR` / `WORK_DIR` / `LABELS` defaults — sourced by every job |
| `src/build_manifest.py` | Scan data → `(image_path, mask_path)` manifest CSV |
| `src/seg_finetune.py` | Fine-tune pretrained FCN-ResNet50 → 2-class segmenter (checkpoint/resume) |
| `src/seg_infer.py` | Segment images → masks; shardable by job array |
| `src/morphology.py` | Masks → area + circularity (`skimage.regionprops`) |
| `src/classify.py` | Morphology + survey labels → RandomForest quality classifier (GroupKFold by organoid) |
| `src/evaluate.py` | Classified results → confusion matrix, balanced accuracy / TNR, feature-importance + per-day plots |
| `slurm/01_interactive.md` | Log in + interactive GPU walkthrough |
| `slurm/02_finetune.sbatch` | Batch GPU fine-tune job |
| `slurm/03_monitor.md` | `squeue`/`sacct`/`nvidia-smi` cheat-sheet |
| `slurm/04_infer_array.sbatch` | Parallel inference via **job array** |
| `slurm/05_morphology_dep.sbatch` | Downstream job via **SLURM dependency** |
| `slurm/06_checkpoint.sbatch` | Preemption + checkpoint/resume |
| `WORKSHOP_GUIDE.md` | Which file teaches which workshop session |

## Quick start (on the cluster)

```bash
conda env create -f environment.yml && conda activate workshop-hpc
mkdir -p logs
# Fine-tune, then run the parallel inference → analysis chain:
fid=$(sbatch --parsable slurm/02_finetune.sbatch)
aid=$(sbatch --parsable --dependency=afterok:$fid slurm/04_infer_array.sbatch)
sbatch --dependency=afterok:$aid slurm/05_morphology_dep.sbatch
```

## Data

Defaults to the DSI path in `config/paths.sh`
(`/net/projects2/promega/.../raw/local/input`), expecting:

```
input/images/raw_images/*.tif               # raw full-res organoid images (best-Z)
input/masks/manual/*.tif                      # hand-drawn ground-truth masks (training labels)
input/json/image_mapping_thresholded_and_manual.json   # pairs image <-> mask
```

`build_manifest.py` reads the json to pair each raw image (`Best Z Filename`)
with its manual mask (`MT Mask Path`); 2,091 image/mask pairs resolve cleanly.
`--mode train` emits image+mask for the fine-tune (real human ground truth);
`--mode infer` emits images only, for the segmentation pass. The model resizes
images and masks together to 384x512 internally, so the full-res TIFs (image and
mask share dimensions) need no pre-resizing. **Predicted masks are an output**
(written by `seg_infer.py`), never an input.

### Survey labels (for `classify.py`)

The classification step is supervised on the **human quality verdict**
(`Acceptable` / `Not Acceptable`). Data-prep scripts (in `workshop-hpc-data/`,
run once when building the shareable dataset) produce these:

- `add_survey_labels.py --propagate` — tallies survey votes per organoid and
  writes `survey_label` into the mapping JSON. The verdict is the majority of
  the organoid's **latest rated timepoint** (Dy30 when present); `--propagate`
  copies that final verdict onto the organoid's earlier-day images too.
- `make_labels.py` — extracts `labels.csv` (`image_id,label,organoid_id`) from
  the labelled JSON; `config/paths.sh` points `LABELS` at it.

Because labels are propagated across an organoid's days, `classify.py` groups
its cross-validation by `organoid_id` so the same organoid can't leak across the
train/test split.

## Manual step-by-step execution

```bash
cd workshop-hpc

# 0. Environment (once)
conda env create -f environment.yml
conda activate workshop-hpc
source config/paths.sh          # sets INPUT_DIR, WORK_DIR, LABELS

# 1. Training manifest: raw images + manual ground-truth masks
python src/build_manifest.py --input-dir "$INPUT_DIR" --mode train \
    --limit 1000 --out "$WORK_DIR/data/manifest_train.csv"

# 2. Fine-tune (GPU) -> $WORK_DIR/runs/seg/seg_finetune_model_final.pt
python src/seg_finetune.py --manifest "$WORK_DIR/data/manifest_train.csv" \
    --out "$WORK_DIR/runs/seg" --epochs 10 --resume

# 3. Inference manifest: raw images only
python src/build_manifest.py --input-dir "$INPUT_DIR" --mode infer \
    --limit 1000 --out "$WORK_DIR/data/manifest_infer.csv"

# 4. Segment images (GPU) -> $WORK_DIR/runs/masks/*_predmask.png
python src/seg_infer.py --manifest "$WORK_DIR/data/manifest_infer.csv" \
    --model "$WORK_DIR/runs/seg/seg_finetune_model_final.pt" --out-dir "$WORK_DIR/runs/masks"

# 5. Masks -> area + circularity
python src/morphology.py --mask-dir "$WORK_DIR/runs/masks" \
    --out "$WORK_DIR/runs/morphology.csv"

# 6. Features + survey labels -> supervised quality classifier
python src/classify.py --features "$WORK_DIR/runs/morphology.csv" \
    --labels "$LABELS" --out "$WORK_DIR/runs/classified.csv"

# 7. Evaluate + plot -> confusion matrix, balanced accuracy/TNR, feature importance
python src/evaluate.py --classified "$WORK_DIR/runs/classified.csv" \
    --out-dir "$WORK_DIR/runs/eval"
```

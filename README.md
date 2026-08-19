# tiled_detection_probe

Answers one question before committing to a QLoRA fine-tune: **can the
BASE (non-fine-tuned) Qwen3-VL-4B-Instruct already detect tiny defects
(V_Crack, Micro_Crack) if it's given native-resolution tiles instead of a
resized whole image?**

Earlier work in this project established that tiny defects vanish under
whole-image resizing (a Micro_Crack box shrinks from 37x48px to 5x7px at
a 1280px resize target) -- but that's a resolution problem, not
necessarily a "the model doesn't know what a crack looks like" problem.
If a tightly-cropped, native-resolution tile already lets the *prompted*
base model find it, fine-tuning may not be needed for detection at all
(at most, for output-format/false-positive cleanup). If it still misses
it even when well-framed, that's real evidence fine-tuning is necessary.

Standalone -- does not depend on or import from `sagemaker_finetuning/`
or any other folder in this repo. `data/` is a copy of that project's
held-out test split only (29 images, ~35MB, not the training images).

## Tile geometry

640x640 tiles, 20% overlap, sliding grid with edge-snapping -- the same
parameters already proven by the RT-DETR pipeline
(`rtdetr_pipeline/training/tile_dataset.py` /
`rtdetr_infer_tiled.py`), not a new guess.

## Two modes

- **`--mode gt-centered`** (default, cheap): for every ground-truth box
  in the selected test images, crops ONE tile centered on it and asks
  the model to detect defects in that tile. A handful of model calls per
  image. Answers "can the model find it when well-framed at all" --
  the right first, cheap check.
- **`--mode full-grid`** (expensive, the real end-to-end test): tiles
  the WHOLE image on the sliding grid (the same geometry real tiled
  inference would use), runs every tile, maps detections back to
  full-image coordinates, dedupes overlaps, then reports real
  Precision/Recall/F1. Slow -- a 9000x4600 image tiles into ~150-190
  windows, each a separate model call -- but it's the only mode that
  also shows the false-positive rate on clean/background regions.

Run `gt-centered` first. Only spend the time/GPU-hours on `full-grid` if
`gt-centered` looks promising.

## What to detect: --label (required)

Neither script has a built-in default prompt -- every run says what it's
looking for via one or more `--label NAME[:DESCRIPTION]` flags (the
description is optional). This mirrors production's labeling-studio
auto_label lambda (`index.py::_build_prompt(task, labels, is_claude)`):
the category list is a parameter, not fixed text, so `--label` builds a
matching system+user prompt pair automatically (`build_labels_prompt()`
in `probe_tiled_base.py`) -- no hand-editing prompt strings, and no risk
of the system and user turns describing different things.

```bash
--label Dead_Cell
--label "Dead_Cell:A solar cell that appears completely dark in an EL image..."
--label Dead_Cell --label Micro_Crack --label V_Crack   # repeatable, multiple categories in one pass
```

## Usage

Requires a GPU instance (e.g. `ml.g4dn.2xlarge`, same as
`sagemaker_finetuning/`). Upload this whole folder on its own.

```bash
pip install -q -U "git+https://github.com/huggingface/transformers"  # Qwen3-VL not in a released version yet
pip install -r requirements.txt

cd scripts

# Cheap first check -- focus on the two known-hard classes
python3 probe_tiled_base.py --limit 5 --focus-classes V_Crack Micro_Crack --label V_Crack --label Micro_Crack

# Reproduce the full 6-class taxonomy, any class
python3 probe_tiled_base.py --limit 5 \
  --label Dead_Cell --label Micro_Crack --label Grid_Defect --label Scratch --label V_Crack --label X_Crack

# The expensive, real end-to-end test (start with 1-2 images)
python3 probe_tiled_base.py --mode full-grid --limit 2 --label V_Crack --label Micro_Crack
```

## Testing on your own image (any GPU box, not just SageMaker)

`scripts/infer_tiled_base.py` doesn't need ground truth at all -- give it
any image, it tiles it natively, runs the base model over every tile,
maps detections back to full-image coordinates, dedupes, and saves an
annotated copy with boxes drawn on the original image:

```bash
cd scripts
python3 infer_tiled_base.py --image /path/to/any_image.jpg --label Dead_Cell --label V_Crack
```

No SageMaker- or vendor-specific code anywhere in this folder -- it only
needs `torch.cuda.is_available()`. Runs the same way on a local/on-prem
GPU box (e.g. a 12GB RTX 3060) as on a SageMaker T4 notebook instance.
The 4-bit-quantized 4B model needs only ~2.5-3GB of weights, well inside
a 12GB card, and this is inference-only (no training, no gradients), so
memory pressure is much lower than the fine-tuning pipeline's.

Output defaults to `<image_stem>_tiled_detected.jpg` next to the input;
override with `--output`. `--conf` filters low-confidence detections
before drawing.

## Reading the output

`gt-centered` prints `FOUND`/`MISS` per ground-truth box plus a per-class
summary at the end (`N/M found`). `full-grid` prints per-image tile
progress plus a final TP/FP/FN/Precision/Recall/F1 table, overall and
per class.

## What this does NOT tell you

- If results are poor, it's worth trying a different `--label` description
  before concluding fine-tuning is required -- a bad prompt and a bad
  model produce the same symptom, and there's no default prompt here to
  fall back on for comparison, only whatever `--label` you passed.
- `gt-centered` mode can't tell you the false-positive rate on clean
  background -- only `full-grid` mode does that.

#!/usr/bin/env python3
"""Tests whether the BASE (non-fine-tuned) Qwen3-VL-4B-Instruct can detect
tiny defects (V_Crack, Micro_Crack) when given native-resolution tiles
instead of a resized whole image -- run this BEFORE committing to a QLoRA
fine-tune. If tiling alone already finds them, fine-tuning may not be
needed for detection at all (at most, for output-format/false-positive
cleanup). If it still misses them, that's real evidence a fine-tune is
necessary -- not just a resize-resolution artifact.

Uses the same tile geometry already proven by the RT-DETR pipeline
(rtdetr_pipeline/training/tile_dataset.py / rtdetr_infer_tiled.py):
640x640 tiles, 20% overlap, sliding grid with edge-snapping.

Fully standalone: no imports from sagemaker_finetuning/ or any other
folder in this repo -- everything this script needs (tiling geometry,
IoU, JSON parsing, the system prompt) is defined here or in this folder's
own utils/. See ../data/ for the ground truth this reads (a copy of
sagemaker_finetuning's held-out test split, 29 images).

Two modes:

  --mode gt-centered (default, cheap): for every ground-truth box in the
      selected test images, crops ONE tile centered on it and asks the
      model to detect defects in that tile. A handful of model calls per
      image -- the right first, cheap check of "can the model find it
      when well-framed at all."

  --mode full-grid (expensive, the real end-to-end test): tiles the WHOLE
      image on a sliding grid (the same geometry real tiled inference
      would use), runs every tile through the model, maps detections back
      to full-image coordinates, dedupes overlaps, then reports real
      Precision/Recall/F1/AP. Slow -- a 9000x4600 image tiles into
      ~150-190 windows, each a separate model.generate() call -- but it's
      the only mode that also shows the false-positive rate on
      clean/background regions, which gt-centered mode can't.

What to detect is always given via --label (repeatable), e.g.:
    python3 probe_tiled_base.py --limit 5 \\
        --label "Dead_Cell:A solar cell that appears completely dark..." \\
        --label V_Crack --label Micro_Crack --label Grid_Defect --label Scratch --label X_Crack
    python3 probe_tiled_base.py --limit 5 --label V_Crack --label Micro_Crack --focus-classes V_Crack Micro_Crack
    python3 probe_tiled_base.py --mode full-grid --limit 2 --label V_Crack --label Micro_Crack
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import load_config, resolve_path
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_label_args(raw_labels: list[str] | None) -> list[dict]:
    """Parses repeated --label NAME[:DESCRIPTION] values into
    build_labels_prompt()'s expected [{"name", "description"}, ...]
    shape. Splits on the FIRST colon only, so a description containing
    its own colons (e.g. "e.g.: like this") doesn't get mangled."""
    if not raw_labels:
        raise SystemExit("at least one --label is required, e.g. --label Dead_Cell")
    labels = []
    for raw in raw_labels:
        name, _, description = raw.partition(":")
        name = name.strip()
        if not name:
            raise SystemExit(f"invalid --label value: {raw!r} -- expected NAME or NAME:DESCRIPTION")
        labels.append({"name": name, "description": description.strip() or None})
    return labels


def build_labels_prompt(labels: list[dict]) -> tuple[str, str]:
    """Builds a (system_prompt, user_text) pair scoped to an arbitrary
    set of labels, each {"name": str, "description": str | None} -- the
    same structural idea as production's labeling-studio auto_label
    lambda (index.py::_build_prompt(task, labels, is_claude)): categories
    are a parameter threaded into both the system prompt's category list
    AND the user-turn text, so the two can't drift out of sync the way
    hand-editing two separate prompt strings risks. This is the only
    prompt-building path in this folder -- no separate fixed/default
    prompt exists, every run must say what it's looking for via --label.

    Reuses this project's existing JSON output schema
    ({"defects": [...]}), not production's raw-array schema -- keeps
    compatibility with everything else here that already expects that
    shape (dedup_by_class, threshold_metrics, etc.).

    Also borrows two things from production's prompt: an explicit "never
    invent new labels" rule and a confidence floor. Both matter more here
    when the category set is narrow (maybe just one label) -- nothing
    else stops the model from reporting something it notices under a
    made-up label string, or reporting a weak/uncertain guess as a full
    detection.
    """
    category_lines = [
        f'- "{l["name"]}": {l["description"]}' if l.get("description") else f'- "{l["name"]}"'
        for l in labels
    ]
    category_list = "\n".join(category_lines)
    label_names = ", ".join(f'"{l["name"]}"' for l in labels)

    system_prompt = f"""You are an expert in solar panel electroluminescence image inspection.

Detect every visible instance of the categories below. Choose the label only from these categories -- never invent new labels. If nothing matches, return an empty defects list.

Possible defects:

{category_list}

Return ONLY valid JSON.

Output schema:

{{
  "defects":[
      {{
          "type":"",
          "bbox":[x1,y1,x2,y2],
          "confidence":0.0
      }}
  ]
}}

Bounding boxes must use original image pixel coordinates. Do not annotate anything with confidence below 0.4.

Do not explain your answer."""

    user_text = (
        f"Process each of these categories one by one for a detection task: {label_names}. "
        "Understand each, search the whole image carefully, and annotate every instance found "
        "with a confidence score. Return the final JSON object of annotations."
    )
    return system_prompt, user_text


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_model_json(raw_text: str) -> dict:
    cleaned = _CODE_FENCE_RE.sub("", raw_text).strip()
    return json.loads(cleaned)


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def tile_grid(img_w: int, img_h: int, tile_size: int, overlap_frac: float) -> list[tuple[int, int, int, int]]:
    """Same geometry as rtdetr_pipeline/training/tile_dataset.py's
    tile_grid() -- sliding window with edge-snapping so the grid always
    covers the full image, never overshoots it."""
    step = int(tile_size * (1 - overlap_frac))
    xs = list(range(0, max(img_w - tile_size, 0) + 1, step)) or [0]
    ys = list(range(0, max(img_h - tile_size, 0) + 1, step)) or [0]
    if xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)
    if ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)))
    return tiles


def dedup_by_class(detections: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """IoU-greedy dedup, grouped by defect type first, since two
    different defect types legitimately overlapping shouldn't suppress
    each other."""
    by_class: dict[str, list[dict]] = {}
    for d in detections:
        by_class.setdefault(d.get("type", "?"), []).append(d)

    kept: list[dict] = []
    for cls_detections in by_class.values():
        cls_kept: list[dict] = []
        for det in sorted(cls_detections, key=lambda d: d.get("confidence", 0.0), reverse=True):
            if not any(compute_iou(det["bbox"], k["bbox"]) > iou_threshold for k in cls_kept):
                cls_kept.append(det)
        kept.extend(cls_kept)
    return kept


def load_ground_truth(test_jsonl: Path, images_dir: Path) -> list[dict]:
    """Parses data/jsonl/test.jsonl's bedrock-conversation format into
    {image_key, image_path, defects} records, in ORIGINAL image pixel
    coordinates."""
    records = []
    for line in test_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        user_content = raw["messages"][0]["content"]
        image_uri = next(c["image"]["source"]["s3Location"]["uri"] for c in user_content if "image" in c)
        image_key = Path(image_uri).name
        image_path = images_dir / image_key
        if not image_path.exists():
            logger.warning("Ground truth references %s but it's not at %s -- skipping", image_key, image_path)
            continue
        assistant_text = raw["messages"][1]["content"][0]["text"]
        defects = json.loads(assistant_text)["defects"]
        records.append({"image_key": image_key, "image_path": image_path, "defects": defects})
    return records


def run_tile_inference(model, processor, tile_image, system_prompt: str, user_text: str, max_new_tokens: int) -> list[dict]:
    import torch

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image", "image": tile_image}, {"type": "text", "text": user_text}]},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        return parse_model_json(output_text).get("defects", [])
    except json.JSONDecodeError:
        return []


def tile_window_for_box(box: list[float], img_w: int, img_h: int, tile_size: int) -> tuple[int, int, int, int]:
    """A tile_size x tile_size window centered on box, clamped to image
    bounds. If box itself is larger than tile_size (e.g. some Dead_Cell/
    X_Crack boxes), the window expands to box size + padding instead of
    cropping the defect out."""
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    eff_w = max(tile_size, int(box_w) + 64)
    eff_h = max(tile_size, int(box_h) + 64)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    tx1 = int(max(0, min(cx - eff_w / 2, img_w - eff_w)))
    ty1 = int(max(0, min(cy - eff_h / 2, img_h - eff_h)))
    tx2 = min(img_w, tx1 + eff_w)
    ty2 = min(img_h, ty1 + eff_h)
    return tx1, ty1, tx2, ty2


def gt_centered_probe(records, model, processor, system_prompt, user_text, max_new_tokens, tile_size, iou_threshold):
    from PIL import Image

    found = 0
    total = 0
    per_class: dict[str, dict[str, int]] = {}

    for rec in records:
        image = Image.open(rec["image_path"]).convert("RGB")
        img_w, img_h = image.size
        print(f"\n=== {rec['image_key']} ({img_w}x{img_h}) ===")

        for gt in rec["defects"]:
            total += 1
            cls = gt["type"]
            per_class.setdefault(cls, {"found": 0, "total": 0})
            per_class[cls]["total"] += 1

            tx1, ty1, tx2, ty2 = tile_window_for_box(gt["bbox"], img_w, img_h, tile_size)
            tile = image.crop((tx1, ty1, tx2, ty2))
            preds = run_tile_inference(model, processor, tile, system_prompt, user_text, max_new_tokens)

            mapped = [
                {"type": p.get("type"), "bbox": [p["bbox"][0] + tx1, p["bbox"][1] + ty1, p["bbox"][2] + tx1, p["bbox"][3] + ty1], "confidence": p.get("confidence")}
                for p in preds if p.get("bbox")
            ]
            best = max((m for m in mapped if m["type"] == cls), key=lambda m: compute_iou(m["bbox"], gt["bbox"]), default=None)
            best_iou = compute_iou(best["bbox"], gt["bbox"]) if best else 0.0

            hit = best_iou >= iou_threshold
            if hit:
                found += 1
                per_class[cls]["found"] += 1
            status = "FOUND" if hit else "MISS "
            print(f"  [{status}] {cls:<15} gt_bbox={[round(v) for v in gt['bbox']]} tile=({tx1},{ty1},{tx2},{ty2}) best_iou={best_iou:.2f} raw_preds={mapped}")

    print(f"\n=== gt-centered summary (IoU>={iou_threshold}) ===")
    print(f"Overall: {found}/{total} ground-truth defects found when tightly cropped")
    for cls, c in sorted(per_class.items()):
        print(f"  {cls:<15} {c['found']}/{c['total']}")


def threshold_metrics(gt_by_image, preds_by_image, iou_threshold):
    tp = fp = fn = 0
    per_class: dict[str, dict[str, int]] = {}

    for image_key, gts in gt_by_image.items():
        preds = sorted(preds_by_image.get(image_key, []), key=lambda p: p.get("confidence", 1.0), reverse=True)
        matched_gt: set[int] = set()

        for pred in preds:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if j in matched_gt or gt["type"] != pred.get("type"):
                    continue
                iou = compute_iou(pred.get("bbox", [0, 0, 0, 0]), gt["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            cls = pred.get("type", "?")
            per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
            if best_iou >= iou_threshold:
                matched_gt.add(best_j)
                tp += 1
                per_class[cls]["tp"] += 1
            else:
                fp += 1
                per_class[cls]["fp"] += 1

        for j, gt in enumerate(gts):
            if j not in matched_gt:
                fn += 1
                cls = gt["type"]
                per_class.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0})
                per_class[cls]["fn"] += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "per_class": per_class}


def full_grid_probe(records, model, processor, system_prompt, user_text, max_new_tokens, tile_size, overlap_frac, iou_threshold):
    from PIL import Image

    gt_by_image: dict[str, list[dict]] = {}
    preds_by_image: dict[str, list[dict]] = {}

    for rec in records:
        image = Image.open(rec["image_path"]).convert("RGB")
        img_w, img_h = image.size
        gt_by_image[rec["image_key"]] = rec["defects"]

        tiles = tile_grid(img_w, img_h, tile_size, overlap_frac)
        print(f"\n=== {rec['image_key']} ({img_w}x{img_h}) -- {len(tiles)} tiles ===")

        all_detections = []
        for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            tile = image.crop((tx1, ty1, tx2, ty2))
            preds = run_tile_inference(model, processor, tile, system_prompt, user_text, max_new_tokens)
            for p in preds:
                if not p.get("bbox"):
                    continue
                all_detections.append({
                    "type": p.get("type"),
                    "bbox": [p["bbox"][0] + tx1, p["bbox"][1] + ty1, p["bbox"][2] + tx1, p["bbox"][3] + ty1],
                    "confidence": p.get("confidence", 1.0),
                })
            if (i + 1) % 20 == 0 or i + 1 == len(tiles):
                print(f"  tile {i + 1}/{len(tiles)} -- {len(all_detections)} raw detection(s) so far")

        deduped = dedup_by_class(all_detections)
        print(f"  raw: {len(all_detections)} -> deduped: {len(deduped)}")
        preds_by_image[rec["image_key"]] = deduped

    metrics = threshold_metrics(gt_by_image, preds_by_image, iou_threshold)

    print(f"\n=== full-grid summary (IoU>={iou_threshold}) ===")
    print(f"TP: {metrics['tp']}   FP: {metrics['fp']}   FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.3f}   Recall: {metrics['recall']:.3f}   F1: {metrics['f1']:.3f}")
    print("\nPer-class:")
    for cls, c in sorted(metrics["per_class"].items()):
        p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
        print(f"  {cls:<15} TP={c['tp']:<4} FP={c['fp']:<4} FN={c['fn']:<4} P={p:.3f} R={r:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["gt-centered", "full-grid"], default="gt-centered")
    parser.add_argument("--limit", type=int, default=5, help="Max number of test images to probe")
    parser.add_argument("--focus-classes", nargs="+", default=None, help="Only include test images containing at least one GT box of these class(es), e.g. V_Crack Micro_Crack")
    parser.add_argument("--tile-size", type=int, default=None, help="Default: configs/config.yaml -> tiling.tile_size")
    parser.add_argument("--overlap-frac", type=float, default=None, help="full-grid mode only. Default: configs/config.yaml -> tiling.overlap_frac")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Tiles have at most a couple defects -- kept low to keep per-tile latency down, esp. for full-grid mode")
    parser.add_argument(
        "--label", action="append", dest="labels", required=True, metavar="NAME[:DESCRIPTION]",
        help="What to detect -- repeatable for multiple categories, e.g. --label \"Dead_Cell:A solar "
             "cell that appears completely dark...\" --label Micro_Crack. Required: this is the only "
             "way this script builds a prompt (see build_labels_prompt()). To reproduce the old "
             "full-6-class behavior, pass all 6: --label Dead_Cell --label Micro_Crack --label "
             "Grid_Defect --label Scratch --label V_Crack --label X_Crack.",
    )
    parser.add_argument("--config", default=None, type=Path)
    args = parser.parse_args()

    labels = parse_label_args(args.labels)
    config = load_config(args.config) if args.config else load_config()
    tile_size = args.tile_size or config["tiling"]["tile_size"]
    overlap_frac = args.overlap_frac if args.overlap_frac is not None else config["tiling"]["overlap_frac"]

    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible -- run this on a GPU-enabled SageMaker notebook instance.")

    test_jsonl = resolve_path(config["data"]["test_jsonl"])
    images_dir = resolve_path(config["data"]["test_images_dir"])
    records = load_ground_truth(test_jsonl, images_dir)
    logger.info("Loaded ground truth for %d test image(s)", len(records))

    if args.focus_classes:
        focus = set(args.focus_classes)
        records = [r for r in records if any(d["type"] in focus for d in r["defects"])]
        logger.info("Filtered to %d image(s) containing %s", len(records), sorted(focus))

    records = records[: args.limit]
    if not records:
        raise SystemExit("No test images matched -- check --focus-classes against actual class names in the dataset.")
    logger.info("Probing %d image(s): %s", len(records), [r["image_key"] for r in records])

    logger.info("Loading BASE model (no LoRA adapter, no fine-tuning): %s", config["model"]["base_model"])
    processor = AutoProcessor.from_pretrained(config["model"]["base_model"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        config["model"]["base_model"], quantization_config=bnb_config, device_map="auto", attn_implementation="sdpa",
    )
    model.eval()

    system_prompt, user_text = build_labels_prompt(labels)
    logger.info("Built prompt for %d label(s): %s", len(labels), [l["name"] for l in labels])

    if args.mode == "gt-centered":
        gt_centered_probe(records, model, processor, system_prompt, user_text, args.max_new_tokens, tile_size, args.iou_threshold)
    else:
        full_grid_probe(records, model, processor, system_prompt, user_text, args.max_new_tokens, tile_size, overlap_frac, args.iou_threshold)


if __name__ == "__main__":
    main()

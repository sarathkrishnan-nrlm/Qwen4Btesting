#!/usr/bin/env python3
"""Runs the BASE (non-fine-tuned) Qwen3-VL-4B-Instruct on any image you
give it, native-resolution TILED (never resized) -- crops it into
overlapping 640x640 tiles, prompts each tile independently, maps every
detection back to full-image coordinates, dedupes overlaps across
adjacent tiles, and saves an annotated copy of the ORIGINAL image with
the final boxes drawn on it.

This is the "give me an image, get back an image with boxes" counterpart
to probe_tiled_base.py's ground-truth-driven modes -- use it to eyeball
results on any image, not just the 29-image held-out test set this
folder ships with. No GPU-vendor-specific code here -- just needs
torch.cuda.is_available(); works the same on a T4 SageMaker notebook or
a local/on-prem GPU box (e.g. RTX 3060 12GB -- a 4-bit-quantized 4B
model needs only ~2.5-3GB of weights, comfortably inside that).

Shares tiling/dedup/prompt logic with probe_tiled_base.py (same folder,
imported directly -- both scripts stay self-contained as a unit, no
dependency outside this folder).

Usage:
    python3 infer_tiled_base.py --image /path/to/any_image.jpg --label Dead_Cell --label V_Crack

    # description narrows what the model looks for within a label
    python3 infer_tiled_base.py --image /path/to/any_image.jpg \\
        --label "Dead_Cell:A solar cell that appears completely dark in an EL image..."

    # custom output path (default: alongside input, named <stem>_tiled_detected.jpg)
    python3 infer_tiled_base.py --image /path/to/any_image.jpg --label V_Crack --output /path/to/result.jpg

    # only keep detections at/above this confidence (model's own reported
    # confidence -- not independently calibrated, treat as a rough filter)
    python3 infer_tiled_base.py --image /path/to/any_image.jpg --label V_Crack --conf 0.3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe_tiled_base import build_labels_prompt, dedup_by_class, parse_label_args, run_tile_inference, tile_grid
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def draw_detections(base_image, defects: list[dict]):
    """Draws each defect's box + label onto a copy of base_image, in
    whatever coordinate space defects' "bbox" is already in -- here,
    always full original-image coordinates (tiles were only a detection
    strategy, not something the caller needs to think about)."""
    from PIL import ImageDraw, ImageFont

    image = base_image.copy()
    draw = ImageDraw.Draw(image)

    line_width = max(2, round(min(image.size) / 400))
    font_size = max(16, round(min(image.size) / 80))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for d in defects:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
        conf = d.get("confidence")
        label = f"{d.get('type', '?')} {conf:.2f}" if isinstance(conf, (int, float)) else str(d.get("type", "?"))
        text_y = max(0, y1 - font_size - 4)
        text_bbox = draw.textbbox((x1, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 0, 0))
        draw.text((x1, text_y), label, fill=(255, 255, 255), font=font)

    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Path to any image -- run at native resolution, never resized")
    parser.add_argument("--output", default=None, type=Path, help="Path to save the annotated image (default: alongside --image, named <stem>_tiled_detected.jpg)")
    parser.add_argument("--tile-size", type=int, default=None, help="Default: configs/config.yaml -> tiling.tile_size")
    parser.add_argument("--overlap-frac", type=float, default=None, help="Default: configs/config.yaml -> tiling.overlap_frac")
    parser.add_argument("--conf", type=float, default=0.0, help="Drop detections below this confidence before drawing/saving")
    parser.add_argument("--dedup-iou", type=float, default=0.4, help="IoU threshold for suppressing duplicate detections across overlapping tiles")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument(
        "--label", action="append", dest="labels", required=True, metavar="NAME[:DESCRIPTION]",
        help="What to detect -- repeatable for multiple categories in one pass, e.g. "
             "--label \"Dead_Cell:A solar cell that appears completely dark...\" --label Micro_Crack. "
             "Required: this is the only way this script builds a prompt (see probe_tiled_base.py's "
             "build_labels_prompt()).",
    )
    parser.add_argument("--config", default=None, type=Path)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")

    labels = parse_label_args(args.labels)

    config = load_config(args.config) if args.config else load_config()
    tile_size = args.tile_size or config["tiling"]["tile_size"]
    overlap_frac = args.overlap_frac if args.overlap_frac is not None else config["tiling"]["overlap_frac"]

    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible -- check `nvidia-smi` and that this environment's torch build has CUDA support.")
    logger.info("GPU: %s", torch.cuda.get_device_name(0))

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

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size
    tiles = tile_grid(img_w, img_h, tile_size, overlap_frac)
    logger.info("%s: %dx%d -> %d tiles (size=%d, overlap=%.2f)", args.image.name, img_w, img_h, len(tiles), tile_size, overlap_frac)

    all_detections = []
    start = time.time()
    for i, (tx1, ty1, tx2, ty2) in enumerate(tiles):
        tile = image.crop((tx1, ty1, tx2, ty2))
        preds = run_tile_inference(model, processor, tile, system_prompt, user_text, args.max_new_tokens)
        for p in preds:
            if not p.get("bbox"):
                continue
            all_detections.append({
                "type": p.get("type"),
                "bbox": [p["bbox"][0] + tx1, p["bbox"][1] + ty1, p["bbox"][2] + tx1, p["bbox"][3] + ty1],
                "confidence": p.get("confidence", 1.0),
            })
        if (i + 1) % 10 == 0 or i + 1 == len(tiles):
            elapsed = time.time() - start
            remaining = (elapsed / (i + 1)) * (len(tiles) - i - 1)
            logger.info("tile %d/%d -- %d raw detection(s) so far -- ~%.0fs remaining", i + 1, len(tiles), len(all_detections), remaining)

    logger.info("Raw detections across all tiles: %d", len(all_detections))
    deduped = dedup_by_class(all_detections, iou_threshold=args.dedup_iou)
    final = [d for d in deduped if d.get("confidence", 1.0) >= args.conf]
    logger.info("After dedup + confidence>=%.2f filter: %d", args.conf, len(final))

    print(f"\n{len(final)} defect(s) detected (original-image pixel coordinates):")
    print(f"  {'Type':<15} {'Bbox':<30} {'Confidence'}")
    for d in sorted(final, key=lambda d: d.get("confidence", 0.0), reverse=True):
        conf = d.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        print(f"  {d.get('type', '?'):<15} {str([round(v) for v in d['bbox']]):<30} {conf_str}")

    if final:
        annotated = draw_detections(image, final)
        output_path = args.output or args.image.with_name(f"{args.image.stem}_tiled_detected.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path, format="JPEG", quality=95)
        print(f"\nAnnotated image saved to: {output_path} (original resolution {img_w}x{img_h} -- native-resolution tiled detection, no resize)")
    else:
        print("\nNo defects detected -- nothing to draw, no annotated image saved.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 4 (vKITTI2 domain gap): Precision / Recall / F1 / mAP@50 of the clone-trained YOLO on one folder.

Ground truth always comes from <out_dir>/test_<condition>/labels/ (boxes converted with the fixed 1242x375
vKITTI2 resolution); predictions use conf 0.25, imgsz 640. Predictions are matched greedily by confidence at
IoU 0.5; AP@50 is 11-point interpolated over the pooled predictions.

Built-in pipelines read the folders written by the other scripts:
    baseline  <out_dir>/test_<condition>/images     (raw degraded frames)
    clahe     <out_dir>/clahe_<condition>/images
    relit     <out_dir>/relit_<condition>/images
Any other method (e.g. an external low-light enhancer) is scored by giving its output folder; its files must
keep the test-set names (<Scene>_f<frame>.jpg/.png):

    python eval/vkitti2/evaluate.py --condition fog --pipeline relit --weights best.pt
    python eval/vkitti2/evaluate.py --condition fog --pipeline other_method --images_dir /path/to/enhanced_fog

Writes <out_dir>/results/<condition>_<pipeline>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vkitti_common import DEFAULT_OUT, PIPELINE_DIRS, list_images

CONF_THRESH = 0.25
IOU_THRESH = 0.5  # IoU threshold for TP matching and AP@50
IMG_SIZE = 640
GT_W, GT_H = 1242, 375  # vKITTI2 native resolution, used to de-normalise the GT labels

DEFAULT_WEIGHTS = DEFAULT_OUT / "runs" / "clone_only" / "weights" / "best.pt"


def iou(box_a, box_b):
    """IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_label_to_xyxy(cx, cy, w, h, img_w, img_h):
    return [(cx - w / 2) * img_w, (cy - h / 2) * img_h, (cx + w / 2) * img_w, (cy + h / 2) * img_h]


def compute_metrics(model, img_dir, lbl_dir, condition, pipeline, limit=None):
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {lbl_dir}")
    img_paths = list_images(img_dir)[:limit]
    if not img_paths:
        raise ValueError(f"No images found in {img_dir}")

    all_tp = all_fp = all_fn = 0
    conf_scores = []    # for AP computation
    matched_flags = []  # TP = 1, FP = 0

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = list(map(float, line.split()))
            if len(parts) == 5:
                _, cx, cy, w, h = parts
                gt_boxes.append(yolo_label_to_xyxy(cx, cy, w, h, GT_W, GT_H))

        preds = model.predict(str(img_path), conf=CONF_THRESH, imgsz=IMG_SIZE, verbose=False)[0].boxes
        pred_boxes, pred_confs = [], []
        if preds is not None and len(preds) > 0:
            for box in preds:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                pred_boxes.append([x1, y1, x2, y2])
                pred_confs.append(float(box.conf[0].cpu()))

        # Greedy matching, highest confidence first
        gt_matched = [False] * len(gt_boxes)
        for i in np.argsort(pred_confs)[::-1]:
            best_iou = IOU_THRESH - 1e-6
            best_j = -1
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                iou_val = iou(pred_boxes[i], gt)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_j = j
            if best_j >= 0:
                gt_matched[best_j] = True
                all_tp += 1
                matched_flags.append(1)
            else:
                all_fp += 1
                matched_flags.append(0)
            conf_scores.append(pred_confs[i])
        all_fn += sum(1 for m in gt_matched if not m)

    precision = all_tp / (all_tp + all_fp + 1e-9)
    recall = all_tp / (all_tp + all_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    # AP@50: 11-point interpolated area under the pooled PR curve
    if conf_scores:
        sorted_pairs = sorted(zip(conf_scores, matched_flags), reverse=True)
        tp_cum = fp_cum = 0
        precisions, recalls = [], []
        total_gt = all_tp + all_fn
        for _, match in sorted_pairs:
            if match:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / (total_gt + 1e-9))
        ap = 0.0
        for thr in np.linspace(0, 1, 11):
            prec_at_thr = [p for p, r in zip(precisions, recalls) if r >= thr]
            ap += (max(prec_at_thr) if prec_at_thr else 0.0) / 11
    else:
        ap = 0.0

    return {
        "condition": condition,
        "pipeline": pipeline,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mAP50": round(ap, 4),
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "num_images": len(img_paths),
        "images_dir": str(img_dir.resolve()),
    }


def resolve_images_dir(out_dir, condition, pipeline, images_dir=None):
    if images_dir:
        return Path(str(images_dir).format(cond=condition))
    if pipeline not in PIPELINE_DIRS:
        raise ValueError(f"pipeline '{pipeline}' is not built in ({', '.join(PIPELINE_DIRS)}); pass --images_dir")
    return Path(out_dir) / PIPELINE_DIRS[pipeline].format(cond=condition)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, help="e.g. fog, rain, sunset, overcast")
    p.add_argument("--pipeline", required=True,
                   help="baseline | clahe | relit, or any name together with --images_dir")
    p.add_argument("--images_dir", default=None,
                   help="Folder of images to score (overrides the pipeline default; '{cond}' is substituted)")
    p.add_argument("--labels_dir", default=None, help="Default <out_dir>/test_<condition>/labels")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Clone-trained YOLO weights")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--limit", type=int, default=None, help="Only the first N images (smoke tests)")
    args = p.parse_args()

    from ultralytics import YOLO

    img_dir = resolve_images_dir(args.out_dir, args.condition, args.pipeline, args.images_dir)
    lbl_dir = Path(args.labels_dir or Path(args.out_dir) / f"test_{args.condition}" / "labels")
    metrics = compute_metrics(YOLO(args.weights), img_dir, lbl_dir, args.condition, args.pipeline, args.limit)
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.out_dir) / "results" / f"{args.condition}_{args.pipeline}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] Saved -> {out_path}")


if __name__ == "__main__":
    main()

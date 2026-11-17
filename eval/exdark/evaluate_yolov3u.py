#!/usr/bin/env python3
"""Step 5 (ExDark): per-class AP@50 and mAP@50 of a trained detector on the relit val split.

Runs ultralytics validation on the `val` split of data.yaml and prints AP@50 for the 12 classes and
their mean (mAP@50). mAP@50-95 is reported alongside.

    python eval/exdark/evaluate_yolov3u.py \
        --weights eval/exdark/outputs/runs/train/yolov3u_relit/weights/best.pt \
        --data eval/exdark/outputs/yolo_relit_aligned/data.yaml

Writes <out_dir>/<name>.json and ultralytics plots to <project>/<name>/.
"""

import argparse
import json
import os

from exdark_common import CLASSES, DEFAULT_OUT


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_OUT / "runs" / "train" / "yolov3u_relit" / "weights" / "best.pt"))
    p.add_argument("--data", default=str(DEFAULT_OUT / "yolo_relit_aligned" / "data.yaml"))
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(DEFAULT_OUT / "runs" / "val"))
    p.add_argument("--name", default="yolov3u_relit_eval")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT / "results"), help="Where the JSON summary goes")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=os.path.abspath(args.data), split="val", project=os.path.abspath(args.project),
                        name=args.name, device=args.device, exist_ok=True, amp=False)

    names = model.names
    box = metrics.box
    ap50 = {names[int(c)]: float(box.ap50[i]) for i, c in enumerate(box.ap_class_index)}
    ap50_95 = {names[int(c)]: float(box.ap[i]) for i, c in enumerate(box.ap_class_index)}

    print("\nPer-class AP@50:")
    print("  ".join(f"{c[:7]:>7}" for c in CLASSES + ["Mean"]))
    print("  ".join(f"{ap50[c]:7.3f}" if c in ap50 else f"{'-':>7}" for c in CLASSES) + f"  {box.map50:7.3f}")
    print(f"mAP@50 = {box.map50:.4f}   mAP@50-95 = {box.map:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out, "w") as f:
        json.dump({"weights": os.path.abspath(args.weights), "data": os.path.abspath(args.data),
                   "ap50": ap50, "mAP50": float(box.map50), "ap50_95": ap50_95, "mAP50_95": float(box.map)},
                  f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

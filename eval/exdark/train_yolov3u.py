#!/usr/bin/env python3
"""Step 4 (ExDark): train YOLOv3u on the relit training split.

Defaults: COCO-pretrained yolov3u.pt, 30 epochs, batch 128,
imgsz 608 (fixed, no multi-scale), SGD lr0 0.001 / lrf 0.01 / momentum 0.9 / weight decay 0.0005, linear LR
schedule with 3 warm-up epochs, mosaic 1.0 closed for the last 10 epochs, seed 0, deterministic, AMP,
NMS conf 0.005 / IoU 0.45 for the per-epoch validation; all other hyper-parameters at ultralytics defaults.

    python eval/exdark/train_yolov3u.py --data eval/exdark/outputs/yolo_relit_aligned/data.yaml

Weights: <project>/<name>/weights/best.pt (default eval/exdark/outputs/runs/train/yolov3u_relit/).
"""

import argparse
import os

from exdark_common import DEFAULT_OUT

# Fixed training settings (not exposed as flags).
TRAIN_ARGS = dict(
    pretrained=True,
    optimizer="SGD",
    lr0=0.001,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    cos_lr=False,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    mosaic=1.0,
    close_mosaic=10,
    multi_scale=False,
    seed=0,
    deterministic=True,
    amp=True,
    iou=0.45,
    conf=0.005,
    exist_ok=True,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_OUT / "yolo_relit_aligned" / "data.yaml"))
    p.add_argument("--model", default="yolov3u.pt", help="Pretrained checkpoint (ultralytics name or path)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--imgsz", type=int, default=608)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="ram", help="ram (default), disk, or none")
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(DEFAULT_OUT / "runs" / "train"))
    p.add_argument("--name", default="yolov3u_relit")
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=False if args.cache == "none" else args.cache,
        device=args.device,
        project=os.path.abspath(args.project),
        name=args.name,
        **TRAIN_ARGS,
    )
    print(f"Weights: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

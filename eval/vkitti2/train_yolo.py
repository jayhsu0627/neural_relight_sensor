#!/usr/bin/env python3
"""Step 2 (vKITTI2 domain gap): fine-tune YOLO26s on the clone (daylight) split only.

Defaults: COCO-pretrained yolo26s.pt, 50 epochs, batch 16,
imgsz 640, optimizer 'auto', lr0 0.001 / lrf 0.01 / weight decay 0.0005, cosine LR, seed 42, deterministic,
AMP, mosaic 0.5 closed for the last 10 epochs, and no photometric augmentation (hsv_h = hsv_s = hsv_v = 0,
flipud 0) so that the detector stays lighting-naive. Everything else is at ultralytics defaults.

    python eval/vkitti2/train_yolo.py --data eval/vkitti2/outputs/yolo_dataset/dataset.yaml

Weights: <project>/<name>/weights/best.pt (default eval/vkitti2/outputs/runs/clone_only/).
"""

import argparse
import os

from vkitti_common import DEFAULT_OUT

TRAIN_ARGS = dict(
    pretrained=True,
    optimizer="auto",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    cos_lr=True,
    seed=42,
    deterministic=True,
    amp=True,
    # No augmentation that changes lighting: the detector must stay lighting-naive.
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    flipud=0.0,
    mosaic=0.5,
    close_mosaic=10,
    verbose=True,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DEFAULT_OUT / "yolo_dataset" / "dataset.yaml"))
    p.add_argument("--model", default="yolo26s.pt", help="Pretrained checkpoint (ultralytics name or path)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default=str(DEFAULT_OUT / "runs"))
    p.add_argument("--name", default="clone_only")
    p.add_argument("--exist_ok", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    data = os.path.abspath(args.data)
    model = YOLO(args.model)
    model.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                project=os.path.abspath(args.project), name=args.name, exist_ok=args.exist_ok, **TRAIN_ARGS)

    best = model.trainer.save_dir / "weights" / "best.pt"
    if best.exists():  # validate the best checkpoint on clone/val, as the original script did
        YOLO(str(best)).val(data=data, imgsz=args.imgsz, device=args.device)
    print(f"Weights: {best}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 1 (vKITTI2 domain gap): parse vKITTI2 bbox.txt into single-class ('Car') YOLO datasets.

  * clone (daylight) -> 80/20 train/val split (random_state 42) for training the detector:
        <out_dir>/yolo_dataset/{train,val}/{images,labels}/, <out_dir>/yolo_dataset/dataset.yaml
  * each test condition -> <out_dir>/test_<condition>/{images,labels}/

Camera_0 only, scenes 01/02/06/18/20; boxes kept if number_pixels >= 100, occupancy_ratio >= 0.3,
width >= 20 px and height >= 10 px. Images are symlinked and named <Scene>_f<frame>.jpg.

    python eval/vkitti2/prepare_vkitti2.py --vkitti_root /data/vkitti2

--vkitti_root must contain vkitti_2.0.3_rgb/ and vkitti_2.0.3_textgt/ (or pass --rgb_root/--textgt_root):
    vkitti_2.0.3_rgb/SceneXX/<condition>/frames/rgb/Camera_0/rgb_XXXXX.jpg
    vkitti_2.0.3_textgt/SceneXX/<condition>/bbox.txt
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from vkitti_common import DEFAULT_OUT, CONDITIONS

SCENES = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
IMG_W, IMG_H = 1242, 375  # vKITTI2 native resolution
CAMERA_ID = 0

# Minimum quality thresholds: skip tiny / mostly occluded boxes
MIN_PIXELS = 100
MIN_OCCUPANCY = 0.3
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 10


def parse_bbox_file(bbox_path):
    """bbox.txt -> filtered DataFrame (empty if the file is missing or unreadable)."""
    try:
        df = pd.read_csv(bbox_path, sep=r"\s+", comment="#")
        df.columns = ["frame", "cameraID", "trackID", "left", "right", "top", "bottom",
                      "number_pixels", "truncation_ratio", "occupancy_ratio", "isMoving"]
        df = df[df["cameraID"] == CAMERA_ID]
        df = df[df["number_pixels"] >= MIN_PIXELS]
        df = df[df["occupancy_ratio"] >= MIN_OCCUPANCY]
        df["w"] = df["right"] - df["left"]
        df["h"] = df["bottom"] - df["top"]
        df = df[df["w"] >= MIN_BOX_WIDTH]
        df = df[df["h"] >= MIN_BOX_HEIGHT]
        return df
    except Exception as e:
        print(f"[WARN] Could not parse {bbox_path}: {e}")
        return pd.DataFrame()


def bbox_to_yolo(left, right, top, bottom, img_w=IMG_W, img_h=IMG_H):
    """Pixel bbox (left, right, top, bottom) -> YOLO (cx, cy, w, h) normalised."""
    cx = ((left + right) / 2) / img_w
    cy = ((top + bottom) / 2) / img_h
    w = (right - left) / img_w
    h = (bottom - top) / img_h
    return cx, cy, w, h


def collect_condition_frames(rgb_root, textgt_root, condition):
    """[{scene, frame, img_path, boxes}] for one condition; boxes are (left, right, top, bottom)."""
    records = []
    for scene in SCENES:
        df = parse_bbox_file(textgt_root / scene / condition / "bbox.txt")
        if df.empty:
            continue
        for frame_id, group in df.groupby("frame"):
            img_path = (rgb_root / scene / condition / "frames" / "rgb" / f"Camera_{CAMERA_ID}"
                        / f"rgb_{int(frame_id):05d}.jpg")
            if not img_path.exists():
                continue
            records.append({
                "scene": scene,
                "frame": int(frame_id),
                "img_path": str(img_path),
                "boxes": list(zip(group["left"], group["right"], group["top"], group["bottom"])),
            })
    return records


def write_yolo_dataset(records, out_dir, class_id=0):
    """Symlink images to out_dir/images/ and write YOLO labels to out_dir/labels/."""
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        stem = f"{rec['scene']}_f{rec['frame']:05d}"
        dst_img = img_dir / f"{stem}.jpg"
        if not dst_img.exists():
            src = Path(rec["img_path"]).resolve()
            try:
                dst_img.symlink_to(src)
            except FileExistsError:
                pass
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst_img)
        lines = []
        for (l, r, t, b) in rec["boxes"]:
            cx, cy, w, h = bbox_to_yolo(l, r, t, b)
            # Clamp to [0, 1]: boxes occasionally extend slightly outside the image
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))


def prepare_clone_trainval(rgb_root, textgt_root, out_root, train_ratio=0.8, seed=42):
    records = collect_condition_frames(rgb_root, textgt_root, "clone")
    train_recs, val_recs = train_test_split(records, train_size=train_ratio, random_state=seed)
    print(f"[clone] {len(records)} frames -> train {len(train_recs)}, val {len(val_recs)}")
    root = out_root / "yolo_dataset"
    write_yolo_dataset(train_recs, root / "train")
    write_yolo_dataset(val_recs, root / "val")
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(f"path: {root.resolve()}\n"
                         f"train: train/images\n"
                         f"val:   val/images\n"
                         f"nc: 1\n"
                         f"names: ['Car']\n")
    print(f"[clone] wrote {yaml_path}")


def prepare_test_condition(rgb_root, textgt_root, out_root, condition):
    records = collect_condition_frames(rgb_root, textgt_root, condition)
    out_dir = out_root / f"test_{condition}"
    write_yolo_dataset(records, out_dir)
    print(f"[{condition}] {len(records)} frames -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vkitti_root", default=None, help="Folder containing vkitti_2.0.3_rgb/ and vkitti_2.0.3_textgt/")
    p.add_argument("--rgb_root", default=None, help="Override <vkitti_root>/vkitti_2.0.3_rgb")
    p.add_argument("--textgt_root", default=None, help="Override <vkitti_root>/vkitti_2.0.3_textgt")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, help="Test conditions to prepare")
    p.add_argument("--skip_trainval", action="store_true", help="Do not (re)build the clone train/val split")
    args = p.parse_args()

    if not (args.vkitti_root or (args.rgb_root and args.textgt_root)):
        p.error("pass --vkitti_root, or both --rgb_root and --textgt_root")
    rgb_root = Path(args.rgb_root or Path(args.vkitti_root) / "vkitti_2.0.3_rgb")
    textgt_root = Path(args.textgt_root or Path(args.vkitti_root) / "vkitti_2.0.3_textgt")
    out_root = Path(args.out_dir)

    if not args.skip_trainval:
        prepare_clone_trainval(rgb_root, textgt_root, out_root)
    for cond in args.conditions:
        prepare_test_condition(rgb_root, textgt_root, out_root, cond)


if __name__ == "__main__":
    main()

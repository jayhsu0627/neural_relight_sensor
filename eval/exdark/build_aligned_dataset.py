#!/usr/bin/env python3
"""Step 3 (ExDark): build a 12-class YOLO dataset aligned to the official train/val split.

Only images present in the relit folders are included. Images are symlinked; labels are written from the
ExDark bbGt annotations, normalised by the size of the relit image (run resize_relit_to_original.py first).

Output layout:
    <out_dir>/images/{train,val}/   symlinks to the relit images
    <out_dir>/labels/{train,val}/   YOLO labels (class cx cy w h, normalised)
    <out_dir>/data.yaml

    python eval/exdark/build_aligned_dataset.py --exdark_root /data/ExDark_all

The same script builds a dataset from any other set of images named like the ExDark originals (e.g. raw
images or another enhancement method) by pointing --image_dirs at them.
"""

import argparse
import glob
import os

from PIL import Image
from tqdm import tqdm

from exdark_common import CLASS_TO_ID, CLASSES, DEFAULT_OUT, index_annotations, load_split


def build_stem_to_image(image_dirs):
    mapping = {}
    for d in image_dirs:
        for fpath in glob.glob(os.path.join(d, "*")):
            if fpath.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                mapping[os.path.splitext(os.path.basename(fpath))[0]] = fpath
    return mapping


def convert_box_to_yolo(img_w, img_h, x, y, w, h):
    """Absolute (x_top_left, y_top_left, w, h) -> YOLO normalised (cx, cy, w, h)."""
    return (x + w / 2.0) / img_w, (y + h / 2.0) / img_h, w / img_w, h / img_h


def parse_anno(anno_path, img_w, img_h):
    """bbGt annotation file -> list of YOLO label lines."""
    lines_out = []
    with open(anno_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if parts[0] not in CLASS_TO_ID:
                continue
            try:
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cx, cy, nw, nh = convert_box_to_yolo(img_w, img_h, x, y, w, h)
                lines_out.append(f"{CLASS_TO_ID[parts[0]]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            except (ValueError, IndexError):
                continue
    return lines_out


def build_split(out_dir, split, stems, stem_to_image, stem_to_anno):
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    processed = no_image = no_anno = 0
    for s in tqdm(sorted(stems), desc=f"Building {split}"):
        image_path = stem_to_image.get(s)
        if image_path is None:
            no_image += 1
            continue
        anno_path = stem_to_anno.get(s)
        if anno_path is None:
            no_anno += 1
            continue

        dst_img = os.path.join(img_dir, os.path.basename(image_path))
        if os.path.islink(dst_img) or os.path.exists(dst_img):
            os.remove(dst_img)
        os.symlink(os.path.abspath(image_path), dst_img)

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            print(f"  Warning: could not read {image_path}: {e}")
            continue

        with open(os.path.join(lbl_dir, s + ".txt"), "w") as f:
            f.write("\n".join(parse_anno(anno_path, img_w, img_h)))
        processed += 1

    print(f"  [{split}] processed={processed}, no_image={no_image}, no_anno={no_anno}")
    return processed


def write_yaml(out_dir, n_train, n_val):
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(out_dir, 'images/train')}\n"
                f"val:   {os.path.join(out_dir, 'images/val')}\n\n"
                f"nc: {len(CLASSES)}\n"
                f"names: {CLASSES}\n\n"
                f"# Split sizes (images with both an image and an annotation available)\n"
                f"# train: {n_train}\n"
                f"# val:   {n_val}\n")
    print(f"Wrote {yaml_path}")
    return yaml_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exdark_root", required=True, help="ExDark root (annotations under ExDark_Annno/<Class>/)")
    p.add_argument("--image_dirs", nargs="+",
                   default=[str(DEFAULT_OUT / "relit" / "train"), str(DEFAULT_OUT / "relit" / "val")],
                   help="Folders with the (relit) images; any split, matched to the lists by file stem")
    p.add_argument("--train_list", default=None, help="Override <root>/EXDark_V2/Exdark/main/train.txt")
    p.add_argument("--val_list", default=None, help="Override <root>/EXDark_V2/Exdark/main/val.txt")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT / "yolo_relit_aligned"))
    args = p.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    train = load_split(args.exdark_root, "train", args.train_list)
    val = load_split(args.exdark_root, "val", args.val_list)
    print(f"train list: {len(train)}  val list: {len(val)}")

    stem_to_image = build_stem_to_image(args.image_dirs)
    stem_to_anno = index_annotations(args.exdark_root)
    print(f"{len(stem_to_image)} images, {len(stem_to_anno)} annotation files indexed")

    os.makedirs(out_dir, exist_ok=True)
    n_train = build_split(out_dir, "train", train, stem_to_image, stem_to_anno)
    n_val = build_split(out_dir, "val", val, stem_to_image, stem_to_anno)
    write_yaml(out_dir, n_train, n_val)
    print(f"Done: {out_dir} (train {n_train}, val {n_val})")


if __name__ == "__main__":
    main()

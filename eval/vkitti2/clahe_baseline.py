#!/usr/bin/env python3
"""Step 3a (vKITTI2 domain gap): CLAHE baseline.

CLAHE on the L channel of LAB (clipLimit 2.0, 8x8 tiles), written with cv2.imwrite under the same file name.
Reads <out_dir>/test_<condition>/images/, writes <out_dir>/clahe_<condition>/images/. Resume-safe.

    python eval/vkitti2/clahe_baseline.py
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from vkitti_common import DEFAULT_OUT, CONDITIONS, list_images

CLIP_LIMIT = 2.0
TILE_GRID = (8, 8)


def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def clahe_folder(out_root, condition):
    src_dir = out_root / f"test_{condition}" / "images"
    dst_dir = out_root / f"clahe_{condition}" / "images"
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(src_dir)
    print(f"[clahe] {condition}: {len(paths)} images")
    for img_path in tqdm(paths, desc=f"CLAHE {condition}"):
        out_path = dst_dir / img_path.name
        if out_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(out_path), apply_clahe(img))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT), help="Folder written by prepare_vkitti2.py")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    args = p.parse_args()
    for cond in args.conditions:
        clahe_folder(Path(args.out_dir), cond)


if __name__ == "__main__":
    main()

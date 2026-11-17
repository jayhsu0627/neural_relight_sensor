#!/usr/bin/env python3
"""Step 2 (ExDark): resize the relit images back to the size of their originals, in place.

The bbGt boxes are in original-image pixels, so the relit images must have the original size before
build_aligned_dataset.py normalises the boxes. Matching is by case-insensitive file stem. Resampling is
LANCZOS; the image is re-saved in its own format (JPEG with quality=95, optimize=True).

    python eval/exdark/resize_relit_to_original.py --exdark_root /data/ExDark_all
"""

import argparse
import os

from PIL import Image
from tqdm import tqdm

from exdark_common import DEFAULT_OUT, IMAGE_EXTS, index_images


def resize_image_to_target(image_path, target_size, output_path, resample=Image.LANCZOS):
    try:
        with Image.open(image_path) as img:
            resized_img = img.resize(target_size, resample=resample)
            img_format = img.format or "JPEG"
            save_kwargs = {}
            if img_format == "JPEG" or str(output_path).lower().endswith((".jpg", ".jpeg")):
                save_kwargs["quality"] = 95
                save_kwargs["optimize"] = True
            resized_img.save(output_path, format=img_format, **save_kwargs)
            return True
    except Exception as e:
        print(f"Error resizing {image_path}: {e}")
        return False


def process_directory(relit_dir, originals, dry_run=False):
    """originals: {lower-case stem: original image path}."""
    stats = {"total": 0, "resized": 0, "same_size": 0, "missing_original": 0, "errors": 0}
    if not os.path.isdir(relit_dir):
        print(f"skipping {relit_dir}: not a directory")
        return stats
    files = sorted(f for f in os.listdir(relit_dir) if f.lower().endswith(IMAGE_EXTS))
    for name in tqdm(files, desc=os.path.basename(os.path.normpath(relit_dir))):
        stats["total"] += 1
        relit_path = os.path.join(relit_dir, name)
        original_path = originals.get(os.path.splitext(name)[0].lower())
        if original_path is None:
            stats["missing_original"] += 1
            continue
        try:
            with Image.open(original_path) as orig_img:
                target_size = orig_img.size
            with Image.open(relit_path) as relit_img:
                relit_size = relit_img.size
            if relit_size == target_size:
                stats["same_size"] += 1
                continue
            if dry_run or resize_image_to_target(relit_path, target_size, relit_path):
                stats["resized"] += 1
            else:
                stats["errors"] += 1
        except Exception as e:
            stats["errors"] += 1
            print(f"Error on {name}: {e}")
    print(f"{relit_dir}: {stats}")
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exdark_root", required=True, help="ExDark root (original images under ExDark/<Class>/)")
    p.add_argument("--relit_dirs", nargs="+",
                   default=[str(DEFAULT_OUT / "relit" / "train"), str(DEFAULT_OUT / "relit" / "val")])
    p.add_argument("--dry_run", action="store_true", help="Report what would be resized without writing")
    args = p.parse_args()

    originals = {stem.lower(): str(path) for stem, path in index_images(args.exdark_root).items()}
    for d in args.relit_dirs:
        process_directory(d, originals, args.dry_run)


if __name__ == "__main__":
    main()

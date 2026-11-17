#!/usr/bin/env python3
"""Step 1 (ExDark): relight the images of the official train/val split.

Inference settings:
  * model resolution: long side 1024, short side by aspect ratio, both rounded to a multiple of 8 and
    clamped to [256, 1024] (the input image itself is passed at full resolution);
  * 4 inference steps, guidance 1.0, the same seed (42) for every image;
  * images loaded with diffusers.utils.load_image (EXIF transpose + RGB);
  * output saved at model resolution under the input file name with PIL defaults.
    Run resize_relit_to_original.py afterwards to restore the original image sizes.

Outputs: <out_dir>/<split>/<original file name>. Resume-safe (existing outputs are skipped).

    python eval/exdark/relight_exdark.py --exdark_root /data/ExDark_all
"""

import argparse
import sys
import time
from pathlib import Path

from exdark_common import DEFAULT_OUT, REPO_ROOT, index_images, load_split

sys.path.insert(0, str(REPO_ROOT))
from relight import BASE_MODEL, LORA_DIR, Relighter, load_pipeline  # noqa: E402

# Exact prompt string of the evaluation runs (same words as relight.PROMPT, different line breaks).
EVAL_PROMPT = """
Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh highlights, and no dramatic contrast.
Maintain the original identity of all subjects exactly—preserve facial structure, skin tone, proportions, expressions, hair, clothing, and textures.
Do not alter pose, camera angle, background geometry, or image composition.
Lighting should appear balanced, and studio-neutral, similar to diffuse overcast or a soft lightbox setup.
Ensure consistent exposure across the entire image with realistic depth and subtle shading only where necessary for form.
"""


def model_size(width, height, max_dim=1024, min_dim=256):
    """Width/height fed to the model: long side max_dim, rounded to /8, each side clamped to [min_dim, max_dim]."""
    aspect = width / height
    if aspect >= 1:
        new_w, new_h = max_dim, int(max_dim / aspect)
    else:
        new_w, new_h = int(max_dim * aspect), max_dim
    new_w, new_h = round(new_w / 8) * 8, round(new_h / 8) * 8
    return max(min_dim, min(max_dim, new_w)), max(min_dim, min(max_dim, new_h))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exdark_root", required=True, help="ExDark root (contains ExDark/, ExDark_Annno/, EXDark_V2/)")
    p.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--train_list", default=None, help="Override <root>/EXDark_V2/Exdark/main/train.txt")
    p.add_argument("--val_list", default=None, help="Override <root>/EXDark_V2/Exdark/main/val.txt")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT / "relit"), help="Writes <out_dir>/<split>/")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--lora", default=LORA_DIR, help="LoRA folder or Hub repo id")
    p.add_argument("--max_dim", type=int, default=1024)
    p.add_argument("--min_dim", type=int, default=256)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images of each split (smoke tests)")
    args = p.parse_args()

    from diffusers.utils import load_image

    images = index_images(args.exdark_root)
    lists = {"train": args.train_list, "val": args.val_list}
    jobs = []
    for split in args.splits:
        stems = sorted(load_split(args.exdark_root, split, lists[split]))
        missing = [s for s in stems if s not in images]
        if missing:
            print(f"[{split}] warning: {len(missing)} listed images not found under ExDark/, e.g. {missing[:3]}")
        stems = [s for s in stems if s in images][: args.limit]
        out_dir = Path(args.out_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs += [(images[s], out_dir / images[s].name) for s in stems]
        print(f"[{split}] {len(stems)} images -> {out_dir}")

    todo = [(src, dst) for src, dst in jobs if not dst.exists()]
    print(f"{len(jobs) - len(todo)} already relit, {len(todo)} to go")
    if not todo:
        return

    relighter = Relighter(load_pipeline(args.base_model, args.lora, device=args.device),
                          max_dim=args.max_dim, prompt=EVAL_PROMPT)
    failed = 0
    t0 = time.time()
    for i, (src, dst) in enumerate(todo, 1):
        try:
            image = load_image(str(src))
            size = model_size(*image.size, max_dim=args.max_dim, min_dim=args.min_dim)
            relighter(image, num_inference_steps=args.steps, seed=args.seed, size=size).save(dst)
        except Exception as e:  # keep going, like the original run
            failed += 1
            print(f"failed: {src.name}: {e}")
        if i % 50 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {(time.time() - t0) / i:.2f} s/image")
    print(f"done: {len(todo) - failed} relit, {failed} failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step 3b (vKITTI2 domain gap): relight the prepared test frames of each condition ("Ours").

Inference settings:
  * native resolution: width and height rounded down to a multiple of 8 (input resized with LANCZOS if that
    changes its size; the FLUX.2 pipeline itself snaps to a multiple of 16, so 1242x375 frames come out
    1232x368);
  * 4 inference steps, guidance 1.0, the same seed (42) for every frame;
  * output saved at model resolution as JPEG quality 95 under the input file name. --keep_size resizes them
    back to the input size instead.

Reads <out_dir>/test_<condition>/images/, writes <out_dir>/relit_<condition>/images/. Resume-safe.

    python eval/vkitti2/relight_vkitti2.py
"""

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from vkitti_common import DEFAULT_OUT, CONDITIONS, REPO_ROOT, list_images

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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=str(DEFAULT_OUT), help="Folder written by prepare_vkitti2.py")
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--lora", default=LORA_DIR, help="LoRA folder or Hub repo id")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep_size", action="store_true",
                   help="Resize outputs back to the input size")
    p.add_argument("--limit", type=int, default=None, help="Only the first N frames per condition (smoke tests)")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    jobs = []
    for cond in args.conditions:
        src_dir = out_root / f"test_{cond}" / "images"
        dst_dir = out_root / f"relit_{cond}" / "images"
        dst_dir.mkdir(parents=True, exist_ok=True)
        paths = list_images(src_dir)[: args.limit]
        print(f"[{cond}] {len(paths)} frames: {src_dir} -> {dst_dir}")
        jobs += [(src, dst_dir / src.name) for src in paths]

    todo = [(src, dst) for src, dst in jobs if not dst.exists()]
    print(f"{len(jobs) - len(todo)} already relit, {len(todo)} to go")
    if not todo:
        return

    relighter = Relighter(load_pipeline(args.base_model, args.lora, device=args.device), prompt=EVAL_PROMPT)
    failed = 0
    t0 = time.time()
    for i, (src, dst) in enumerate(todo, 1):
        try:
            img = Image.open(src).convert("RGB")
            input_size = img.size
            width, height = (img.width // 8) * 8, (img.height // 8) * 8
            if img.size != (width, height):
                img = img.resize((width, height), Image.LANCZOS)
            relit = relighter(img, num_inference_steps=args.steps, seed=args.seed, size=(width, height))
            if args.keep_size:
                relit = relit.resize(input_size, Image.LANCZOS)
            relit.save(dst, quality=95)
        except Exception as e:  # keep going, like the original run
            failed += 1
            print(f"failed: {src}: {e}")
        if i % 100 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {(time.time() - t0) / i:.2f} s/frame")
    print(f"done: {len(todo) - failed} relit, {failed} failed")


if __name__ == "__main__":
    main()

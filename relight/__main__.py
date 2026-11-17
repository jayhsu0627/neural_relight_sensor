"""Relight an image or a folder of images.

    python -m relight --input night.png --output relit.png
    python -m relight --input images/ --output relit/ --max_dim 1024 --steps 4 --keep_size
"""

import argparse
import os
import sys

from PIL import Image

from relight.pipeline import BASE_MODEL, LORA_DIR, Relighter, load_pipeline

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Image file or folder")
    p.add_argument("--output", required=True, help="Output file (single image) or folder")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--lora", default=LORA_DIR, help="LoRA folder or Hub repo id")
    p.add_argument("--max_dim", type=int, default=1024, help="Long side fed to the model (CARLA uses 256)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_size", action="store_true", help="Resize the result back to the input size")
    args = p.parse_args()

    if os.path.isdir(args.input):
        names = sorted(f for f in os.listdir(args.input) if f.lower().endswith(EXTS))
        pairs = [(os.path.join(args.input, n), os.path.join(args.output, n)) for n in names]
        os.makedirs(args.output, exist_ok=True)
    else:
        pairs = [(args.input, args.output)]
    if not pairs:
        sys.exit(f"No images found in {args.input}")

    relight = Relighter(load_pipeline(args.base_model, args.lora), max_dim=args.max_dim)
    for i, (src, dst) in enumerate(pairs, 1):
        image = Image.open(src).convert("RGB")
        out = relight(image, num_inference_steps=args.steps, seed=args.seed)
        if args.keep_size:
            out = out.resize(image.size, Image.LANCZOS)
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        out.save(dst)
        print(f"[{i}/{len(pairs)}] {src} -> {dst}")


if __name__ == "__main__":
    main()

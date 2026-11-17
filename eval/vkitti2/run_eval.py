#!/usr/bin/env python3
"""Step 5 (vKITTI2 domain gap): evaluate every condition x pipeline and print the domain-gap table.

Runs evaluate.py's metrics for baseline / CLAHE / external methods / relit (Ours) on fog, rain, overcast and
sunset, caches each result in <out_dir>/results/<condition>_<pipeline>.json (skipped if present unless
--recompute), then prints Precision / Recall / F1 / mAP@50 per condition and method.

    python eval/vkitti2/run_eval.py --weights eval/vkitti2/outputs/runs/clone_only/weights/best.pt \
        --external other_method=/path/to/other_method_outputs/{cond}
"""

import argparse
import json
from pathlib import Path

from evaluate import DEFAULT_WEIGHTS, compute_metrics, resolve_images_dir
from vkitti_common import DEFAULT_OUT, CONDITIONS

def print_full_table(results, conditions, pipelines):
    print("\n" + "=" * 72)
    print(f"{'Condition':<12} {'Pipeline':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'mAP@50':>8}")
    print("=" * 72)
    for cond in conditions:
        for pipe in pipelines:
            r = results.get((cond, pipe))
            if r:
                print(f"{cond:<12} {pipe:<14} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                      f"{r['f1']:>8.3f} {r['mAP50']:>8.3f}")
        print("-" * 72)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Clone-trained YOLO weights")
    p.add_argument("--out_dir", default=str(DEFAULT_OUT))
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--pipelines", nargs="+", default=["baseline", "clahe", "relit"])
    p.add_argument("--external", action="append", default=[], metavar="NAME=DIR",
                   help="Extra method: NAME=folder of its outputs, '{cond}' is replaced by the condition "
                        "(repeatable); skipped for conditions whose folder does not exist")
    p.add_argument("--recompute", action="store_true", help="Ignore cached result JSONs")
    p.add_argument("--table_only", action="store_true", help="Only print tables from cached JSONs")
    p.add_argument("--limit", type=int, default=None, help="Only the first N images per folder (smoke tests)")
    args = p.parse_args()

    out_root = Path(args.out_dir)
    external = dict(e.split("=", 1) for e in args.external)
    # Column order: baseline, CLAHE, external methods, relit.
    pipelines = [x for x in args.pipelines if x != "relit"] + list(external) + \
                (["relit"] if "relit" in args.pipelines else [])

    model = None
    results = {}
    for cond in args.conditions:
        for pipe in pipelines:
            out = out_root / "results" / f"{cond}_{pipe}.json"
            if out.exists() and not args.recompute:
                results[(cond, pipe)] = json.loads(out.read_text())
                continue
            if args.table_only:
                continue
            img_dir = resolve_images_dir(out_root, cond, pipe, external.get(pipe))
            if not img_dir.exists():
                print(f"[skip] {cond} x {pipe}: {img_dir} not found")
                continue
            if model is None:
                from ultralytics import YOLO
                model = YOLO(args.weights)
            print(f"[eval] {cond} x {pipe}: {img_dir}")
            r = compute_metrics(model, img_dir, out_root / f"test_{cond}" / "labels", cond, pipe, args.limit)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(r, indent=2))
            results[(cond, pipe)] = r

    print_full_table(results, args.conditions, pipelines)


if __name__ == "__main__":
    main()

# Detection benchmarks

Scripts to measure how much the neural relight sensor helps a detector on two datasets: ExDark (real low-light
photos) and Virtual KITTI 2 (a daylight-trained detector tested under fog, rain, overcast and sunset). Relighting
uses the `relight` package of this repository: by default FLUX.2-klein-4b with the Delight LoRA in
`weights/delight_lora` (`--base_model` / `--lora` select another model).

Run everything from the repository root. Every input and output path is a flag; outputs default to
`eval/exdark/outputs/` and `eval/vkitti2/outputs/`. The scripts do not download datasets: point them at a local copy.

Requirements: `pip install -r requirements-eval.txt`. The pretrained detectors (`yolov3u.pt`, `yolo26s.pt`) are
fetched by ultralytics on first use. Relighting takes about 1 s per image on one RTX PRO 6000 (4 steps, bf16).

## ExDark (per-class AP@50 and mAP@50, YOLOv3)

Expected layout of `--exdark_root`:

```
ExDark/<Class>/<image>                   7,363 images in 12 class folders
ExDark_Annno/<Class>/<image>.txt         bbGt annotations
EXDark_V2/Exdark/main/train.txt, val.txt official split lists
```

```bash
ROOT=/path/to/ExDark_all

# 1. Relight the train and val images (writes eval/exdark/outputs/relit/{train,val}/, resume-safe)
python eval/exdark/relight_exdark.py --exdark_root $ROOT

# 2. Resize the relit images back to their original sizes (in place; the boxes are in original pixels)
python eval/exdark/resize_relit_to_original.py --exdark_root $ROOT

# 3. Build the 12-class YOLO dataset aligned to the official split
python eval/exdark/build_aligned_dataset.py --exdark_root $ROOT
#    -> eval/exdark/outputs/yolo_relit_aligned/{images,labels}/{train,val}/, data.yaml

# 4. Train YOLOv3u on the relit training images (30 epochs, batch 128, imgsz 608, SGD; see the script)
python eval/exdark/train_yolov3u.py
#    -> eval/exdark/outputs/runs/train/yolov3u_relit/weights/best.pt

# 5. Evaluate on the relit val split
python eval/exdark/evaluate_yolov3u.py
#    -> per-class AP@50 and mAP@50, eval/exdark/outputs/results/yolov3u_relit_eval.json
```

`build_aligned_dataset.py --image_dirs ...` builds the same dataset from any other set of images that keep the
ExDark file names, for example the raw images or another enhancement method, so the comparison uses the same
detector recipe.

## Virtual KITTI 2 (Precision / Recall / F1 / mAP@50, YOLO26s)

The detector is trained on the `clone` (daylight) frames only and tested, without retraining, on the `fog`, `rain`,
`overcast` and `sunset` versions of the same scenes, with raw frames, CLAHE, relit frames, or any external
enhancer. `--vkitti_root` must contain `vkitti_2.0.3_rgb/` and `vkitti_2.0.3_textgt/`.

```bash
VK=/path/to/vkitti2

# 1. Parse bbox.txt into single-class ('Car') YOLO data: clone 80/20 train/val + one test set per condition
python eval/vkitti2/prepare_vkitti2.py --vkitti_root $VK

# 2. Train YOLO26s on clone only (50 epochs, imgsz 640, no photometric augmentation; see the script)
python eval/vkitti2/train_yolo.py
#    -> eval/vkitti2/outputs/runs/clone_only/weights/best.pt

# 3. Enhance the test frames
python eval/vkitti2/clahe_baseline.py       # -> outputs/clahe_<condition>/images/
python eval/vkitti2/relight_vkitti2.py      # -> outputs/relit_<condition>/images/  (resume-safe)

# 4. Evaluate every condition x method
python eval/vkitti2/run_eval.py --external other_method=/path/to/other_method_outputs/{cond}
#    -> outputs/results/<condition>_<method>.json and a P/R/F1/mAP@50 table
```

Single cells can be scored with `evaluate.py`, e.g.
`python eval/vkitti2/evaluate.py --condition fog --pipeline relit`. To compare another enhancer, run it on
`eval/vkitti2/outputs/test_<condition>/images/` (keep the file names and the 1242x375 size) and pass the folder with
`--external NAME=DIR` (`{cond}` is replaced by the condition) to `run_eval.py`, or with `--pipeline NAME --images_dir
DIR` to `evaluate.py`.

Predictions use conf 0.25 and imgsz 640, are matched greedily to the ground truth at IoU 0.5 over all frames of a
condition, and AP@50 is the 11-point interpolated area under that PR curve. `relight_vkitti2.py --keep_size`
resizes the relit frames back to the input size.

"""Shared helpers for the ExDark scripts: class list, dataset indexing and the official split lists.

Expected layout of --exdark_root (the standard ExDark release plus the EXDark_V2 split lists):

    <exdark_root>/ExDark/<Class>/<image>                      images (12 class folders)
    <exdark_root>/ExDark_Annno/<Class>/<image>.txt            bbGt annotations
    <exdark_root>/EXDark_V2/Exdark/main/{train,val}.txt       official 5,896 / 1,467 split lists
"""

import os
from pathlib import Path

CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat",
           "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
CLASS_TO_ID = {cls: i for i, cls in enumerate(CLASSES)}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / "outputs"

def stem(name):
    """'2015_00001.png' -> '2015_00001'; '2015_00001.png.txt' -> '2015_00001'."""
    base = os.path.basename(name)
    if base.endswith(".txt"):
        base = os.path.splitext(base)[0]
    return os.path.splitext(base)[0]


def split_list_path(exdark_root, split):
    return Path(exdark_root) / "EXDark_V2" / "Exdark" / "main" / f"{split}.txt"


def load_stems(txt_path):
    """Set of basenames-without-extension from a split .txt file."""
    stems = set()
    with open(txt_path) as f:
        for line in f:
            name = line.strip()
            if name:
                stems.add(os.path.splitext(os.path.basename(name))[0])
    return stems


def load_split(exdark_root, split, list_path=None):
    return load_stems(list_path or split_list_path(exdark_root, split))


def index_images(exdark_root):
    """{stem: path} for every image under <exdark_root>/ExDark/<Class>/."""
    mapping = {}
    img_root = Path(exdark_root) / "ExDark"
    for cls_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                mapping[stem(f.name)] = f
    return mapping


def index_annotations(exdark_root):
    """{stem: path} for every bbGt file under <exdark_root>/ExDark_Annno/<Class>/ (named like 2015_00001.png.txt)."""
    mapping = {}
    anno_root = Path(exdark_root) / "ExDark_Annno"
    for cls_dir in sorted(p for p in anno_root.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.glob("*.txt")):
            mapping[stem(f.name)] = f
    return mapping

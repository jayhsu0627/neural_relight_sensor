"""Shared constants for the vKITTI2 scripts."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_OUT = HERE / "outputs"

# Degraded test conditions. The detector is trained on `clone` only.
CONDITIONS = ["fog", "rain", "overcast", "sunset"]

# Default image folder of each built-in pipeline, relative to --out_dir (written by the prepare /
# clahe / relight scripts). Any other method is scored with evaluate.py --images_dir.
PIPELINE_DIRS = {
    "baseline": "test_{cond}/images",
    "clahe": "clahe_{cond}/images",
    "relit": "relit_{cond}/images",
}


def list_images(folder):
    folder = Path(folder)
    return sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))

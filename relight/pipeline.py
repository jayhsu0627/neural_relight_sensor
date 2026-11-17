"""FLUX.2-klein-4B + Delight LoRA: relight an image to neutral, uniform illumination."""

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = "black-forest-labs/FLUX.2-klein-4b"
LORA_DIR = os.path.join(REPO_ROOT, "weights", "delight_lora")
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

# The instruction every training pair in linoyts/flux-delight-dataset uses.
PROMPT = (
    "Relight the image to remove all existing lighting conditions and replace them with neutral, "
    "uniform illumination. Apply soft, evenly distributed lighting with no directional shadows, no harsh "
    "highlights, and no dramatic contrast. Maintain the original identity of all subjects exactly—preserve "
    "facial structure, skin tone, proportions, expressions, hair, clothing, and textures. Do not alter pose, "
    "camera angle, background geometry, or image composition. Lighting should appear balanced, and "
    "studio-neutral, similar to diffuse overcast or a soft lightbox setup. Ensure consistent exposure across "
    "the entire image with realistic depth and subtle shading only where necessary for form."
)

ImageLike = Union[Image.Image, np.ndarray]


def load_pipeline(
    base_model: str = BASE_MODEL,
    lora: Optional[str] = LORA_DIR,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    compile_transformer: bool = False,
):
    """Load Flux2KleinPipeline and attach the Delight LoRA (a local directory or a Hub repo id)."""
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(base_model, torch_dtype=dtype)
    if lora:
        pipe.load_lora_weights(lora, weight_name=LORA_WEIGHT_NAME)
    pipe.to(device)
    if compile_transformer:
        # Compile only forward() so the module keeps its attributes. Not compatible with the async sensor.
        pipe.transformer.forward = torch.compile(pipe.transformer.forward, mode="reduce-overhead", fullgraph=False)
    return pipe


def fit_size(width: int, height: int, max_dim: int) -> Tuple[int, int]:
    """Scale so the long side is max_dim, round to multiples of 8, clamp to [max(128, max_dim // 2), max_dim]."""
    aspect = width / height
    if aspect >= 1:
        w, h = max_dim, int(max_dim / aspect)
    else:
        w, h = int(max_dim * aspect), max_dim
    w, h = round(w / 8) * 8, round(h / 8) * 8
    lo = max(128, max_dim // 2)
    return max(lo, min(max_dim, w)), max(lo, min(max_dim, h))


def to_pil(image: ImageLike) -> Image.Image:
    """Accept a PIL image or an RGB uint8 array of shape (H, W, 3)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.ascontiguousarray(image[..., :3]))


class Relighter:
    """Callable wrapper around the pipeline with the paper's inference settings.

    max_dim=256 is what the CARLA loop uses (1024 / 4); use 1024 for offline benchmarks.
    """

    def __init__(self, pipeline=None, max_dim: int = 256, prompt: str = PROMPT, **load_kwargs):
        self.pipeline = pipeline if pipeline is not None else load_pipeline(**load_kwargs)
        self.max_dim = max_dim
        self.prompt = prompt
        self.device = self.pipeline.transformer.device

    @torch.inference_mode()
    def __call__(
        self,
        image: ImageLike,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Relight one image. Returns a PIL image at the model resolution (see fit_size) or `size`."""
        pil = to_pil(image)
        width, height = size if size is not None else fit_size(*pil.size, self.max_dim)
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipeline(
            prompt=self.prompt,
            image=[pil],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

#!/usr/bin/env python
# coding=utf-8
"""
Customized training script for FLUX.2 Klein Delight LoRA using flux-delight-dataset.

This script is based on train_dreambooth_lora_flux2_img2img.py from diffusers
(Copyright The HuggingFace Team, Apache License 2.0), adapted for:
- FLUX.2 Klein models (the paper uses FLUX.2-klein-4b; see train/paper.sh)
- linoyts/flux-delight-dataset structure (input_image, output_image, prompt)
- Delighting/relighting task

Dataset structure:
- input_image: Image with existing lighting conditions (conditioning)
- output_image: Uniformly lit target image (ground truth)
- prompt: Delighting instruction text
"""

import argparse
import copy
import glob
import itertools
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder, hf_hub_download
from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
from peft.tuners.lora.layer import Linear as PeftLinear
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    BitsAndBytesConfig,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.training_utils import (
    _collate_lora_metadata,
    _to_cpu_contiguous,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    find_nearest_bucket,
    free_memory,
    get_fsdp_kwargs_from_accelerator,
    offload_models,
    parse_buckets_string,
    wrap_with_fsdp,
)
try:
    # Prefer importing Blackwell helpers via the local Flux2 scripts directory
    import sys
    import os as _os

    _flux2_dir = _os.path.join(_os.path.dirname(__file__), "..", "Flux2")
    _flux2_dir = _os.path.abspath(_flux2_dir)
    if _flux2_dir not in sys.path:
        sys.path.append(_flux2_dir)

    from blackwell_utils import (
        BlackwellLinear,
        patch_flux2_with_blackwell,
        normalize_checkpoint_keys,
        build_state_dict_for_load,
    )
except Exception:
    # If Flux2 utilities are not available, we fall back to standard transformer loading
    patch_flux2_with_blackwell = None
    normalize_checkpoint_keys = None
    build_state_dict_for_load = None
    # Optional NVFP4 helpers (only used with --use_blackwell_nvfp4; train/paper.sh does not use them).


class BlackwellLoraLinear(PeftLinear):
    """
    Custom LoRA wrapper for BlackwellLinear layers.

    This reuses PEFT's standard Linear LoRA implementation but wraps the NVFP4
    BlackwellLinear base layer. The base path still uses NVFP4 via
    torch._scaled_mm, and the LoRA residual is applied in floating point on top.

    We intentionally disable weight merging for this layer type, since
    BlackwellLinear does not expose a dense `weight` parameter to merge into.
    """

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "Merging LoRA weights into BlackwellLinear is not supported. "
            "Use dynamic LoRA (no merge) for NVFP4 Blackwell layers."
        )

    def unmerge(self) -> None:  # type: ignore[override]
        raise NotImplementedError(
            "Unmerging LoRA weights from BlackwellLinear is not supported because merge is disabled."
        )

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:  # type: ignore[override]
        # Disable conversion-based merging paths (e.g., convert_to_lora / torchao merge).
        return False

from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module


if getattr(torch, "distributed", None) is not None:
    import torch.distributed as dist

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.37.0.dev0")

logger = get_logger(__name__)


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
    fp8_training=False,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux.2 [Klein] Delight DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux2 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md).

This LoRA was specifically trained for delighting/relighting tasks using the `linoyts/flux-delight-dataset`.

FP8 training? {fp8_training}

## Trigger words

You should use the delighting prompt to trigger the image transformation:
"{instance_prompt}"

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import Flux2KleinPipeline
import torch

pipeline = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B", 
    torch_dtype=torch.bfloat16
).to('cuda')

pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')

# Delight an image
image = pipeline(
    prompt="{instance_prompt}",
    image=[input_image],  # Image with lighting
    num_inference_steps=4,
    guidance_scale=1.0,
).images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.2/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "image-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux2-klein",
        "flux2-klein-diffusers",
        "template:sd-lora",
        "delighting",
        "relighting",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def run_yolo_detection(image, yolo_model):
    """
    Run YOLO object detection on an image.
    
    Args:
        image: PIL Image or numpy array (RGB format)
        yolo_model: YOLO model instance
    
    Returns:
        tuple: (annotated_image, num_detections, avg_confidence, detections)
            - annotated_image: numpy array with bounding boxes drawn
            - num_detections: number of objects detected
            - avg_confidence: average confidence score of detections
            - detections: YOLO results object
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if img_array.shape[2] == 4:  # RGBA to RGB
                img_array = img_array[:, :, :3]
        else:
            img_array = image
        
        # Ensure contiguous array for YOLO
        if not img_array.flags['C_CONTIGUOUS']:
            img_array = np.ascontiguousarray(img_array)
        
        # Run YOLO detection
        results = yolo_model(img_array, verbose=False)
        
        # Get annotated image with bounding boxes
        annotated_image = results[0].plot()
        
        # Extract detection statistics
        boxes = results[0].boxes
        num_detections = len(boxes) if boxes is not None else 0
        avg_confidence = float(boxes.conf.mean()) if boxes is not None and len(boxes) > 0 else 0.0
        
        return annotated_image, num_detections, avg_confidence, boxes
    except Exception as e:
        logger.warning(f"YOLO detection failed: {e}")
        return None, 0, 0.0, None


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    args.num_validation_images = args.num_validation_images if args.num_validation_images else 1
    pipeline = pipeline.to(dtype=torch_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    # Load YOLO model if provided
    yolo_model = None
    if args.yolo_model_path is not None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {args.yolo_model_path}")
            yolo_model = YOLO(args.yolo_model_path)
            logger.info("YOLO model loaded successfully")
        except ImportError:
            logger.warning("ultralytics not installed. Install with: pip install ultralytics")
            yolo_model = None
        except Exception as e:
            logger.warning(f"Failed to load YOLO model: {e}")
            yolo_model = None

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    
    # Determine autocast dtype based on accelerator mixed precision
    if accelerator.mixed_precision == "bf16":
        autocast_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None
        
    autocast_ctx = torch.autocast(accelerator.device.type, dtype=autocast_dtype) if not is_final_validation and autocast_dtype is not None else nullcontext()

    images = []
    yolo_results = []  # Store YOLO comparison results
    
    # Check if we have multiple validation images from directory
    if "images" in pipeline_args:
        # Multiple validation images from directory
        validation_images = pipeline_args["images"]
        logger.info(
            f"Running validation... \n Generating images for {len(validation_images)} validation images with prompt:"
            f" {args.validation_prompt}."
        )
        
        for val_img_idx, val_image in enumerate(validation_images):
            # Run YOLO on input image (before relighting)
            input_yolo_result = None
            if yolo_model is not None:
                input_annotated, input_num_det, input_avg_conf, _ = run_yolo_detection(val_image, yolo_model)
                input_yolo_result = {
                    "annotated": input_annotated,
                    "num_detections": input_num_det,
                    "avg_confidence": input_avg_conf,
                }
            
            # Generate num_validation_images samples for each validation image
            for sample_idx in range(args.num_validation_images):
                with autocast_ctx:
                    # Build pipeline call arguments
                    pipeline_call_kwargs = {
                        "image": [val_image],
                        "prompt_embeds": pipeline_args["prompt_embeds"],
                        "generator": generator,
                        "num_inference_steps": 4,  # Explicitly set for better quality
                        "guidance_scale": 3.5,
                    }
                    
                    image = pipeline(**pipeline_call_kwargs).images[0]
                    images.append(image)
                    
                    # Run YOLO on output image (after relighting)
                    if yolo_model is not None:
                        output_annotated, output_num_det, output_avg_conf, _ = run_yolo_detection(image, yolo_model)
                        yolo_results.append({
                            "val_img_idx": val_img_idx,
                            "sample_idx": sample_idx,
                            "input": input_yolo_result,
                            "output": {
                                "annotated": output_annotated,
                                "num_detections": output_num_det,
                                "avg_confidence": output_avg_conf,
                            },
                        })
    elif "image" in pipeline_args:
        # Single validation image - generate num_validation_images samples
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        # Run YOLO on input image (before relighting) if available
        input_yolo_result = None
        if yolo_model is not None and "image" in pipeline_args and len(pipeline_args["image"]) > 0:
            input_image = pipeline_args["image"][0]
            input_annotated, input_num_det, input_avg_conf, _ = run_yolo_detection(input_image, yolo_model)
            input_yolo_result = {
                "annotated": input_annotated,
                "num_detections": input_num_det,
                "avg_confidence": input_avg_conf,
            }
        
        for sample_idx in range(args.num_validation_images):
            with autocast_ctx:
                # Build pipeline call arguments - image is optional (for img2img vs txt2img)
                pipeline_call_kwargs = {
                    "prompt_embeds": pipeline_args["prompt_embeds"],
                    "generator": generator,
                    "num_inference_steps": 4,  # Explicitly set for better quality
                    "guidance_scale": 3.5,
                }
                # Only add image if provided (for image-to-image mode)
                pipeline_call_kwargs["image"] = pipeline_args["image"]
                
                image = pipeline(**pipeline_call_kwargs).images[0]
                images.append(image)
                
                # Run YOLO on output image (after relighting)
                if yolo_model is not None:
                    output_annotated, output_num_det, output_avg_conf, _ = run_yolo_detection(image, yolo_model)
                    yolo_results.append({
                        "sample_idx": sample_idx,
                        "input": input_yolo_result,
                        "output": {
                            "annotated": output_annotated,
                            "num_detections": output_num_det,
                            "avg_confidence": output_avg_conf,
                        },
                    })
    else:
        # Text-to-image validation (no input image)
        logger.info(
            f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
            f" {args.validation_prompt}."
        )
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                # Build pipeline call arguments
                pipeline_call_kwargs = {
                    "prompt_embeds": pipeline_args["prompt_embeds"],
                    "generator": generator,
                    "num_inference_steps": 4,  # Explicitly set for better quality
                    "guidance_scale": 3.5,
                }
                
                image = pipeline(**pipeline_call_kwargs).images[0]
                images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            
            # Log YOLO results to tensorboard if available
            if yolo_model is not None and len(yolo_results) > 0:
                # Log YOLO metrics
                input_detections = [r["input"]["num_detections"] if r["input"] else 0 for r in yolo_results]
                output_detections = [r["output"]["num_detections"] for r in yolo_results]
                input_confidences = [r["input"]["avg_confidence"] if r["input"] else 0.0 for r in yolo_results]
                output_confidences = [r["output"]["avg_confidence"] for r in yolo_results]
                
                tracker.writer.add_scalar(f"{phase_name}/yolo_input_avg_detections", np.mean(input_detections), epoch)
                tracker.writer.add_scalar(f"{phase_name}/yolo_output_avg_detections", np.mean(output_detections), epoch)
                tracker.writer.add_scalar(f"{phase_name}/yolo_input_avg_confidence", np.mean(input_confidences), epoch)
                tracker.writer.add_scalar(f"{phase_name}/yolo_output_avg_confidence", np.mean(output_confidences), epoch)
                tracker.writer.add_scalar(f"{phase_name}/yolo_detection_improvement", np.mean(output_detections) - np.mean(input_detections), epoch)
                
        if tracker.name == "wandb":
            # Create captions based on validation setup
            if "images" in pipeline_args:
                # Multiple validation images - show which input image and sample number
                num_val_images = len(pipeline_args["images"])
                captions = []
                for i, image in enumerate(images):
                    val_img_idx = i // args.num_validation_images
                    sample_idx = i % args.num_validation_images
                    captions.append(f"ValImg{val_img_idx}_Sample{sample_idx}: {args.validation_prompt}")
            else:
                # Single validation image or text-to-image
                captions = [f"{i}: {args.validation_prompt}" for i in range(len(images))]
            
            log_dict = {
                phase_name: [
                    wandb.Image(image, caption=caption) for image, caption in zip(images, captions)
                ]
            }
            
            # Add YOLO results if available
            if yolo_model is not None and len(yolo_results) > 0:
                # Log YOLO metrics
                input_detections = [r["input"]["num_detections"] if r["input"] else 0 for r in yolo_results]
                output_detections = [r["output"]["num_detections"] for r in yolo_results]
                input_confidences = [r["input"]["avg_confidence"] if r["input"] else 0.0 for r in yolo_results]
                output_confidences = [r["output"]["avg_confidence"] for r in yolo_results]
                
                log_dict[f"{phase_name}_yolo_metrics"] = {
                    "input_avg_detections": np.mean(input_detections),
                    "output_avg_detections": np.mean(output_detections),
                    "input_avg_confidence": np.mean(input_confidences),
                    "output_avg_confidence": np.mean(output_confidences),
                    "detection_improvement": np.mean(output_detections) - np.mean(input_detections),
                    "confidence_improvement": np.mean(output_confidences) - np.mean(input_confidences),
                }
                
                # Log YOLO annotated images (input and output)
                yolo_images = []
                yolo_captions = []
                for i, result in enumerate(yolo_results):
                    if result["input"] and result["input"]["annotated"] is not None:
                        yolo_images.append(Image.fromarray(result["input"]["annotated"]))
                        yolo_captions.append(f"Input_YOLO_{i}: {result['input']['num_detections']} detections, conf={result['input']['avg_confidence']:.2f}")
                    
                    if result["output"]["annotated"] is not None:
                        yolo_images.append(Image.fromarray(result["output"]["annotated"]))
                        yolo_captions.append(f"Output_YOLO_{i}: {result['output']['num_detections']} detections, conf={result['output']['avg_confidence']:.2f}")
                
                if len(yolo_images) > 0:
                    log_dict[f"{phase_name}_yolo_detections"] = [
                        wandb.Image(img, caption=caption) for img, caption in zip(yolo_images, yolo_captions)
                    ]
            
            tracker.log(log_dict)

    del pipeline
    free_memory()

    return images


def module_filter_fn(mod: torch.nn.Module, fqn: str):
    # don't convert the output module
    if fqn == "proj_out":
        return False
    # don't convert linear modules with weight dimensions not divisible by 16
    if isinstance(mod, torch.nn.Linear):
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
    return True


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train Flux.2 Klein Delight LoRA using flux-delight-dataset.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4b",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4b",
        help="Path to base model (BF16) for loading Tokenizer, Text Encoder, and VAE if not present in pretrained_model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--bnb_quantization_config_path",
        type=str,
        default=None,
        help="Quantization config in a JSON file that will be used to define the bitsandbytes quant config of the DiT.",
    )
    parser.add_argument(
        "--use_blackwell_nvfp4",
        action="store_true",
        help=(
            "If set, load the transformer using native Blackwell NVFP4 weights instead of BitsAndBytes or "
            "standard fp8/fp16 weights. Expects `--pretrained_model_name_or_path` to point to the NVFP4 repo "
            "(e.g. black-forest-labs/FLUX.2-klein-4b-nvfp4)."
        ),
    )
    parser.add_argument(
        "--do_fp8_training",
        action="store_true",
        help="if we are doing FP8 training.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="linoyts/flux-delight-dataset",
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data. "
            "For flux-delight-dataset, this should be 'linoyts/flux-delight-dataset'."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="output_image",
        help="The column of the dataset containing the target image (uniformly lit).",
    )
    parser.add_argument(
        "--cond_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the conditioning image (with lighting).",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="prompt",
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="A validation image (path) that is used during validation.",
    )
    parser.add_argument(
        "--validation_image_dir",
        type=str,
        default=None,
        help="A directory containing validation images. All images in this directory will be used for validation.",
    )
    parser.add_argument(
        "--yolo_model_path",
        type=str,
        default=None,
        help="Path to YOLO model for object detection evaluation. If provided, will run YOLO on input and output images to compare detection quality.",
    )
    parser.add_argument(
        "--skip_final_inference",
        default=False,
        action="store_true",
        help="Whether to skip the final inference step with loaded lora weights upon training completion.",
    )
    parser.add_argument(
        "--final_validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during a final validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout probability for LoRA layers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-delight-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--dataset_percentage",
        type=float,
        default=1.0,
        help="Percentage of the dataset to use for training (0.0 to 1.0). Useful for fast debugging.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.2 klein variant guidance scale",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoder to CPU when they are not used.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--enable_npu_flash_attention", action="store_true", help="Enabla Flash Attention for NPU")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Specify `--dataset_name`")

    # Check if dataset_name is actually a directory of preprocessed latents
    is_preprocessed_dir = os.path.isdir(args.dataset_name) and any(f.endswith(".pt") for f in os.listdir(args.dataset_name) if os.path.isfile(os.path.join(args.dataset_name, f)))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Pass this info to args for main
    args.is_preprocessed_dataset = is_preprocessed_dir

    return args


class DreamBoothImg2ImgDataset(Dataset):
    """
    A dataset to prepare the instance images with conditioning images and prompts for fine-tuning the model.
    Designed for flux-delight-dataset structure: {input_image, output_image, prompt}
    """

    def __init__(
        self,
        dataset_name,
        dataset_config_name,
        cache_dir,
        image_column,
        cond_image_column,
        caption_column,
        size=1024,
        repeats=1,
        center_crop=False,
        buckets=None,
        dataset_percentage=1.0,
    ):
        self.size = size
        self.center_crop = center_crop
        self.buckets = buckets
        self.dataset_percentage = dataset_percentage

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`."
            )

        # Downloading and loading a dataset from the hub.
        if dataset_name.endswith(".py") and os.path.exists(dataset_name):
            # Load local python script as a module to get the generator
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_dataset", dataset_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, "vkitti_delight_generator"):
                from datasets import Dataset
                print(f"Loading dataset from generator in {dataset_name}...")
                dataset = Dataset.from_generator(module.vkitti_delight_generator)
            else:
                raise ValueError(f"Custom script {dataset_name} must contain 'vkitti_delight_generator' function")
        else:
            dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                cache_dir=cache_dir,
                split="train",
            )
        
        # Subsample dataset if requested
        if self.dataset_percentage < 1.0:
            num_samples = int(len(dataset) * self.dataset_percentage)
            dataset = dataset.select(range(num_samples))
            print(f"Subsampled dataset to {num_samples} samples ({self.dataset_percentage*100}%)")

        # Preprocessing the datasets.
        column_names = dataset.column_names

        # Get the column names for input/target images and captions.
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        if cond_image_column not in column_names:
            raise ValueError(
                f"`--cond_image_column` value '{cond_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

        self.instance_images = dataset[image_column]
        self.cond_images = dataset[cond_image_column]
        self.instance_prompts = dataset[caption_column]

        # Repeat dataset
        self.instance_images = list(itertools.chain.from_iterable(itertools.repeat(x, repeats) for x in self.instance_images))
        self.cond_images = list(itertools.chain.from_iterable(itertools.repeat(x, repeats) for x in self.cond_images))
        self.instance_prompts = list(itertools.chain.from_iterable(itertools.repeat(x, repeats) for x in self.instance_prompts))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images[index % self.num_instance_images]
        cond_image = self.cond_images[index % self.num_instance_images]
        instance_prompt = self.instance_prompts[index % self.num_instance_images]

        # Load images if they are paths (for custom datasets returning paths)
        if isinstance(instance_image, str):
            instance_image = load_image(instance_image)
        if isinstance(cond_image, str):
            cond_image = load_image(cond_image)

        # Process both images
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        cond_image = exif_transpose(cond_image)
        if not cond_image.mode == "RGB":
            cond_image = cond_image.convert("RGB")

        # Find the closest bucket
        if self.buckets is not None:
            width, height = instance_image.size
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            example["bucket_idx"] = bucket_idx
        else:
            target_height, target_width = self.size, self.size
            example["bucket_idx"] = 0

        # Apply transforms
        instance_image = self.train_transform(
            instance_image,
            size=(target_height, target_width),
            center_crop=self.center_crop,
            random_flip=False,  # Don't flip for paired images
        )
        cond_image = self.train_transform(
            cond_image,
            size=(target_height, target_width),
            center_crop=self.center_crop,
            random_flip=False,  # Don't flip for paired images
        )

        example["instance_images"] = instance_image
        example["cond_images"] = cond_image
        example["instance_prompt"] = instance_prompt

        return example

    def train_transform(self, image, size=(224, 224), center_crop=False, random_flip=False):
        # 1. Resize (deterministic)
        resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)

        # 2. Crop: either center or SAME random crop
        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
        else:
            # get_params returns (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)

        # 3. Random horizontal flip with the SAME coin flip
        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)

        # 4. ToTensor + Normalize (deterministic)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))

        return image

class PreprocessedLatentDataset(Dataset):
    """
    Dataset for loading preprocessed .pt files containing VAE latents.
    Loads EVERYTHING into memory for speed (assuming dataset fits in RAM).
    Can optionally move data to GPU if it fits.
    """
    def __init__(self, data_dir, dataset_percentage=1.0, device=None):
        self.data_dir = data_dir
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])
        
        print(f"Loading preprocessed dataset from {data_dir}...")
        self.instance_latents = []
        self.cond_latents = []
        self.instance_prompts = []
        
        # Load all files
        for f_path in tqdm(self.files, desc="Loading latents to RAM"):
            try:
                data = torch.load(f_path, map_location="cpu")
                
                # Handle both batched and single format
                if "instance_params" in data:
                    inst = data["instance_params"]
                    cond = data["cond_params"]
                    prompts = data["prompts"] if "prompts" in data else [data["prompt"]]
                    
                    # Ensure they are batches (add dim if single)
                    if inst.dim() == 3: # (C, H, W)
                        inst = inst.unsqueeze(0)
                        cond = cond.unsqueeze(0)
                        
                    self.instance_latents.append(inst)
                    self.cond_latents.append(cond)
                    self.instance_prompts.extend(prompts)
            except Exception as e:
                print(f"Error loading {f_path}: {e}")

        # Concatenate all into big tensors
        if len(self.instance_latents) > 0:
            self.instance_latents = torch.cat(self.instance_latents, dim=0)
            self.cond_latents = torch.cat(self.cond_latents, dim=0)
        else:
            raise ValueError("No valid data loaded!")

        self.total_samples = len(self.instance_latents)
        print(f"Loaded {self.total_samples} samples.")

        if dataset_percentage < 1.0:
            num_samples = int(self.total_samples * dataset_percentage)
            self.instance_latents = self.instance_latents[:num_samples]
            self.cond_latents = self.cond_latents[:num_samples]
            self.instance_prompts = self.instance_prompts[:num_samples]
            print(f"Subsampled to {num_samples} samples ({dataset_percentage*100}%)")
            self.total_samples = num_samples
            
        # Optimization: Move entire dataset to GPU if possible
        # This eliminates CPU->GPU transfer latency during training
        if device is not None:
            try:
                print(f"Moving {self.total_samples} samples to GPU {device}...")
                # Check memory usage approx: samples * C * H * W * 4 bytes
                # 8600 * 32 * 128 * 128 * 4 = 18GB. 
                # 96GB VRAM should handle 18GB dataset + 4B model (~8GB with grads) + Activations.
                self.instance_latents = self.instance_latents.to(device)
                self.cond_latents = self.cond_latents.to(device)
                print("Dataset moved to GPU successfully.")
            except RuntimeError as e:
                print(f"Failed to move dataset to GPU (OOM likely): {e}. Keeping on CPU.")
            
        self._length = self.total_samples

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {
            "instance_latents_params": self.instance_latents[index], 
            "cond_latents_params": self.cond_latents[index],
            "instance_prompt": self.instance_prompts[index]
        }

def collate_preprocessed(examples):
    instance_params = [example["instance_latents_params"] for example in examples]
    cond_params = [example["cond_latents_params"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    
    instance_params = torch.stack(instance_params) # (B, 2*C, H, W)
    cond_params = torch.stack(cond_params)         # (B, 2*C, H, W)
    
    return {
        "instance_latents_params": instance_params,
        "cond_latents_params": cond_params,
        "prompts": prompts,
        "is_preprocessed": True
    }



def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    cond_pixel_values = [example["cond_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    cond_pixel_values = torch.stack(cond_pixel_values)
    cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "cond_pixel_values": cond_pixel_values, "prompts": prompts}
    return batch


class BucketBatchSampler(BatchSampler):
    def __init__(self, dataset: DreamBoothImg2ImgDataset, batch_size: int, drop_last: bool = False):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got drop_last={}".format(drop_last))

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(self.dataset.buckets) if self.dataset.buckets else [1])]
        for idx in range(len(dataset)):
            example = dataset[idx]
            bucket_idx = example["bucket_idx"]
            self.bucket_indices[bucket_idx].append(idx)

        self.sampler_len = 0
        self.batches = []

        # Pre-generate batches for each bucket
        for indices_in_bucket in self.bucket_indices:
            # Shuffle indices within the bucket
            random.shuffle(indices_in_bucket)
            # Create batches
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip partial batch if drop_last is True
                self.batches.append(batch)
                self.sampler_len += 1

    def __iter__(self):
        # Shuffle the order of the batches each epoch
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.sampler_len


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    if args.do_fp8_training:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # For bf16, don't use native_amp (scaler) since bf16 doesn't need gradient scaling
    # Set mixed_precision_dtype to None to prevent scaler creation
    mixed_precision_dtype = args.mixed_precision
    if args.mixed_precision == "bf16":
        # bf16 doesn't need a scaler, but we still want bf16 autocast
        # We'll handle this by using mixed_precision="bf16" but disabling native_amp
        pass
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision_dtype,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    
    # Disable native AMP scaler for bf16 to avoid FP16 gradient unscaling conflicts
    # bf16 doesn't need gradient scaling like fp16 does
    if args.mixed_precision == "bf16":
        accelerator.native_amp = False
        # Also ensure the scaler is not used by checking after prepare

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    # Use base_model_name_or_path for components that might be missing or quantized in the main model
    components_path = args.base_model_name_or_path if args.base_model_name_or_path else args.pretrained_model_name_or_path
    
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        components_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        components_path,
        subfolder="scheduler",
        revision=args.revision,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Load VAE from base model path to ensure high precision
    vae = AutoencoderKLFlux2.from_pretrained(
        components_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    # Cast BN stats to weight_dtype to match latents dtype after VAE encoding
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=accelerator.device, dtype=weight_dtype)
    latents_bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
        device=accelerator.device, dtype=weight_dtype
    )

    quantization_config = None
    if args.bnb_quantization_config_path is not None:
        with open(args.bnb_quantization_config_path, "r") as f:
            config_kwargs = json.load(f)
            if "load_in_4bit" in config_kwargs and config_kwargs["load_in_4bit"]:
                config_kwargs["bnb_4bit_compute_dtype"] = weight_dtype
        quantization_config = BitsAndBytesConfig(**config_kwargs)

    # Load transformer
    if args.use_blackwell_nvfp4 and patch_flux2_with_blackwell is not None:
        # Native Blackwell NVFP4 path using quantized checkpoint
        logger.info("Loading transformer using native Blackwell NVFP4 integration.")
        # Download NVFP4 checkpoint from the specified repo
        repo_id = args.pretrained_model_name_or_path
        filename = "flux-2-klein-4b-nvfp4.safetensors"
        logger.info(f"Downloading NVFP4 weights {filename} from {repo_id}...")
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"Loading NVFP4 weights from {weight_path}...")
        checkpoint = normalize_checkpoint_keys(load_file(weight_path))

        # Build transformer architecture from base model config
        logger.info(f"Loading transformer config from {components_path}...")
        transformer_config = Flux2Transformer2DModel.load_config(
            components_path,
            subfolder="transformer",
        )
        with torch.device("meta"):
            transformer = Flux2Transformer2DModel.from_config(transformer_config)

        # Materialize on CPU
        transformer = transformer.to_empty(device="cpu")

        # Patch with BlackwellLinear + NVFP4 weights
        logger.info("Patching transformer with Blackwell NVFP4 layers...")
        transformer = patch_flux2_with_blackwell(transformer, checkpoint)

        # Load remaining weights (e.g., norms/biases) if present
        state_dict_for_load = build_state_dict_for_load(checkpoint)
        incompatible = transformer.load_state_dict(state_dict_for_load, strict=False, assign=True)
        logger.info(
            "NVFP4 state loaded. Missing keys: %s | Unexpected keys: %s",
            len(getattr(incompatible, "missing_keys", [])),
            len(getattr(incompatible, "unexpected_keys", [])),
        )
    else:
        # Original path: BitsAndBytes / standard transformer weights
        transformer = None
        try:
            transformer = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
                quantization_config=quantization_config,
                torch_dtype=weight_dtype,
            )
        except (OSError, EnvironmentError) as e:
            logger.warning(f"Could not load transformer directly from {args.pretrained_model_name_or_path}: {e}")
            logger.info(f"Attempting to load base model from {components_path} and swap weights...")
            
            # Load standard base model structure
            transformer = Flux2Transformer2DModel.from_pretrained(
                components_path,
                subfolder="transformer",
                revision=args.revision,
                variant=args.variant,
                quantization_config=quantization_config,
                torch_dtype=weight_dtype,
            )
            
            # Attempt to download and load the weights manually
            # This handles single-file checkpoint repos like the NVFP4 one
            try:
                repo_id = args.pretrained_model_name_or_path
                
                # Heuristic for filename if not standard
                filename = "flux-2-klein-4b-nvfp4.safetensors"
                if "nvfp4" not in repo_id:
                    # Default safetensors name?
                    filename = "diffusion_pytorch_model.safetensors"
                
                logger.info(f"Downloading weights {filename} from {repo_id}...")
                weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
                
                logger.info(f"Loading weights from {weight_path}...")
                state_dict = load_file(weight_path)
                
                # Load into model
                # strict=False allows flexibility if metadata keys differ
                transformer.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded weights.")
                
            except Exception as inner_e:
                logger.error(f"Failed to manually load weights: {inner_e}")
                raise e

    if args.bnb_quantization_config_path is not None:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        components_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder.requires_grad_(False)

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu device ")

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    to_kwargs = {"dtype": weight_dtype, "device": accelerator.device} if not args.offload else {"dtype": weight_dtype}
    # flux vae is stable in bf16 so load it in weight_dtype to reduce memory
    vae.to(**to_kwargs)
    # we never offload the transformer to CPU, so we can just use the accelerator device
    transformer_to_kwargs = (
        {"device": accelerator.device}
        if args.bnb_quantization_config_path is not None
        else {"device": accelerator.device, "dtype": weight_dtype}
    )

    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if not is_fsdp:
        transformer.to(**transformer_to_kwargs)

    if args.do_fp8_training:
        convert_to_float8_training(
            transformer, module_filter_fn=module_filter_fn, config=Float8LinearConfig(pad_inner_dim=True)
        )

    text_encoder.to(**to_kwargs)
    # Initialize a text encoding pipeline and keep it to CPU for now.
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        components_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

    # now we will add new LoRA weights the transformer layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    # If we're using the native Blackwell NVFP4 path, register a custom PEFT
    # module so LoRA can wrap `BlackwellLinear` layers instead of failing with
    # "Target module BlackwellLinear() is not supported".
    if args.use_blackwell_nvfp4 and patch_flux2_with_blackwell is not None and BlackwellLinear is not None:
        try:
            transformer_lora_config._register_custom_module({BlackwellLinear: BlackwellLoraLinear})
            logger.info("Registered custom BlackwellLoRA module for BlackwellLinear layers.")
        except Exception as e:
            logger.warning(f"Failed to register custom BlackwellLoRA module: {e}")

    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        transformer_cls = type(unwrap_model(transformer))

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_model(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        state_dict = accelerator.get_state_dict(model) if is_fsdp else None

        # 3) Only main process materializes the LoRA state dict
        transformer_lora_layers_to_save = None
        if accelerator.is_main_process:
            peft_kwargs = {}
            if is_fsdp:
                peft_kwargs["state_dict"] = state_dict

            transformer_lora_layers_to_save = get_peft_model_state_dict(
                unwrap_model(transformer_model) if is_fsdp else transformer_model,
                **peft_kwargs,
            )

            if is_fsdp:
                transformer_lora_layers_to_save = _to_cpu_contiguous(transformer_lora_layers_to_save)

            # make sure to pop weight so that corresponding model is not saved again
            if weights:
                weights.pop()

            Flux2KleinPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                **_collate_lora_metadata(modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        if not is_fsdp:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
        else:
            transformer_ = Flux2Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="transformer",
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = Flux2KleinPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    # For FP16 mixed precision, cast LoRA to FP32
    # For BF16 mixed precision, keep LoRA in BF16 (don't cast to FP32) to avoid gradient mismatches
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]
    logger.info(f"Using parsed aspect ratio buckets: {buckets}")

    # Dataset and DataLoaders creation:
    if getattr(args, "is_preprocessed_dataset", False):
        logger.info(f"Loading preprocessed latent dataset from {args.dataset_name}")
        train_dataset = PreprocessedLatentDataset(
            data_dir=args.dataset_name,
            dataset_percentage=args.dataset_percentage,
            device=accelerator.device # Move to GPU if possible
        )
        collate_fn_to_use = collate_preprocessed
        # We don't need buckets for preprocessed latents (usually fixed resolution)
        # Or if we did, we'd need metadata. Assuming fixed resolution for now.
    else:
        train_dataset = DreamBoothImg2ImgDataset(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            cache_dir=args.cache_dir,
            image_column=args.image_column,
            cond_image_column=args.cond_image_column,
            caption_column=args.caption_column,
            size=args.resolution,
            repeats=args.repeats,
            center_crop=args.center_crop,
            buckets=buckets,
            dataset_percentage=args.dataset_percentage,
        )
        collate_fn_to_use = collate_fn

    # Sampler handling
    # BucketBatchSampler only works if dataset has buckets
    if getattr(args, "is_preprocessed_dataset", False):
        batch_sampler = None
        shuffle = True
    else:
        batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True)
        shuffle = False # Sampler handles shuffling

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        batch_size=args.train_batch_size if batch_sampler is None else 1,
        shuffle=shuffle,
        collate_fn=collate_fn_to_use,
        num_workers=args.dataloader_num_workers,
        pin_memory=not getattr(args, "is_preprocessed_dataset", False) # Disable pin_memory if data already on GPU
    )

    def compute_text_embeddings(prompt, text_encoding_pipeline):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=args.text_encoder_out_layers,
            )
        return prompt_embeds, text_ids

    # Pre-compute text embeddings for all prompts (they're the same for delighting)
    if hasattr(train_dataset, "instance_prompts") and len(train_dataset.instance_prompts) > 0:
        sample_prompt = train_dataset.instance_prompts[0]
    elif hasattr(train_dataset, "files") and len(train_dataset) > 0:
        # Preprocessed dataset, read first file/batch to get prompt
        temp_item = train_dataset[0]
        sample_prompt = temp_item["instance_prompt"]
    else:
        sample_prompt = ""
        
    with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
        prompt_embeds, text_ids = compute_text_embeddings(sample_prompt, text_encoding_pipeline)

    # Handle validation
    validation_kwargs = {}
    if args.validation_prompt is not None:
        with offload_models(text_encoding_pipeline, device=accelerator.device, offload=args.offload):
            (validation_embeds, validation_text_ids) = compute_text_embeddings(
                args.validation_prompt, text_encoding_pipeline
            )
            validation_kwargs["prompt_embeds"] = validation_embeds
            validation_kwargs["text_ids"] = validation_text_ids

        # Load validation images - either single image or all images from directory
        validation_images = []
        if args.validation_image_dir is not None:
            # Load all images from directory
            image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
            validation_image_paths = []
            for ext in image_extensions:
                validation_image_paths.extend(glob.glob(os.path.join(args.validation_image_dir, ext)))
                validation_image_paths.extend(glob.glob(os.path.join(args.validation_image_dir, ext.upper())))
            
            validation_image_paths = sorted(validation_image_paths)
            logger.info(f"Found {len(validation_image_paths)} validation images in {args.validation_image_dir}")
            
            for img_path in validation_image_paths:
                try:
                    validation_images.append(load_image(img_path))
                except Exception as e:
                    logger.warning(f"Failed to load validation image {img_path}: {e}")
            
            if len(validation_images) > 0:
                validation_kwargs["images"] = validation_images
            else:
                logger.warning(f"No valid images found in {args.validation_image_dir}")
        elif args.validation_image is not None:
            # Single validation image
            validation_image = load_image(args.validation_image)
            validation_kwargs["image"] = [validation_image]

    # Init FSDP for text encoder
    if hasattr(args, "fsdp_text_encoder") and args.fsdp_text_encoder:
        fsdp_kwargs = get_fsdp_kwargs_from_accelerator(accelerator)
        text_encoder_fsdp = wrap_with_fsdp(
            model=text_encoding_pipeline.text_encoder,
            device=accelerator.device,
            offload=args.offload,
            limit_all_gathers=True,
            use_orig_params=True,
            fsdp_kwargs=fsdp_kwargs,
        )

        text_encoding_pipeline.text_encoder = text_encoder_fsdp
        if getattr(torch, "distributed", None) is not None:
            dist.barrier()

    # if cache_latents is set to True, we encode images to latents and store them.
    # If we are using preprocessed dataset, we already have latents!
    is_preprocessed = getattr(args, "is_preprocessed_dataset", False)
    precompute_latents = args.cache_latents and not is_preprocessed
    
    if precompute_latents:
        prompt_embeds_cache = []
        text_ids_cache = []
        latents_cache = []
        cond_latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                with offload_models(vae, device=accelerator.device, offload=args.offload):
                    batch["pixel_values"] = batch["pixel_values"].to(
                        accelerator.device, non_blocking=True, dtype=vae.dtype
                    )
                    batch["cond_pixel_values"] = batch["cond_pixel_values"].to(
                        accelerator.device, non_blocking=True, dtype=vae.dtype
                    )
                    latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                    cond_latents_cache.append(vae.encode(batch["cond_pixel_values"]).latent_dist)

    # move back to cpu before deleting to ensure memory is freed
    if args.cache_latents and not is_preprocessed:
        vae = vae.to("cpu")
        del vae

    # move back to cpu before deleting to ensure memory is freed
    text_encoding_pipeline = text_encoding_pipeline.to("cpu")
    del text_encoder, tokenizer
    free_memory()

    # Scheduler and math around the number of training steps.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * accelerator.num_processes * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # Check loaded transformer precision and structure
    if accelerator.is_main_process:
        print("\n--- Model Inspection ---")
        print(f"Transformer type: {type(unwrap_model(transformer))}")
        # Print first few layers to see dtype
        for name, param in itertools.islice(unwrap_model(transformer).named_parameters(), 5):
            print(f"Param: {name}, Dtype: {param.dtype}, Device: {param.device}")
        
        # Check for 4-bit layers (Linear4bit)
        has_4bit = False
        for module in unwrap_model(transformer).modules():
            if "Linear4bit" in str(type(module)) or "Params4bit" in str(type(module)):
                has_4bit = True
                print("Found 4-bit layer/params:", type(module))
                break
        
        if has_4bit:
            print("Model is using 4-bit quantization (likely BitsAndBytes NF4).")
        else:
            print("Model does NOT appear to be using standard BitsAndBytes 4-bit layers.")
            print("If using NVFP4, weights might be stored in a packed format or custom kernel.")
            
        print("------------------------\n")
    
    # After prepare, check what mixed precision is actually being used
    actual_mixed_precision = accelerator.mixed_precision
    use_bf16_no_scaler = (args.mixed_precision == "bf16" or actual_mixed_precision == "bf16")
    
    # Check if optimizer has a scaler wrapper (indicates fp16 is being used)
    has_scaler_wrapper = False
    if hasattr(optimizer, 'scaler') and optimizer.scaler is not None:
        has_scaler_wrapper = True
    elif hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
        has_scaler_wrapper = True
    
    # Warn if mixed_precision wasn't explicitly set and accelerate is using fp16
    if args.mixed_precision is None:
        if actual_mixed_precision == "fp16":
            logger.warning(
                "Accelerate is using fp16 mixed precision from config. "
                "If you encounter 'unscale FP16 gradients' errors, explicitly set --mixed_precision=bf16 or --mixed_precision=no"
            )
        elif actual_mixed_precision == "bf16":
            logger.info("Using bf16 mixed precision from accelerate config.")
    
    if use_bf16_no_scaler and has_scaler_wrapper:
        logger.warning("bf16 training detected but scaler wrapper exists. Will attempt to unwrap optimizer.")

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-flux2-klein-delight-lora"
        args_cp = vars(args).copy()
        args_cp["text_encoder_out_layers"] = str(args_cp["text_encoder_out_layers"])
        accelerator.init_trackers(tracker_name, config=args_cp)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            prompts = batch["prompts"]

            with accelerator.accumulate(models_to_accumulate):
                # Use pre-computed prompt embeddings (all prompts are the same for delighting)
                num_repeat_elements = len(prompts)
                prompt_embeds_batch = prompt_embeds.repeat(num_repeat_elements, 1, 1)
                text_ids_batch = text_ids.repeat(num_repeat_elements, 1, 1)

                # Convert images to latent space
                if args.cache_latents and not getattr(args, "is_preprocessed_dataset", False):
                    model_input = latents_cache[step].mode().to(dtype=weight_dtype)
                    cond_model_input = cond_latents_cache[step].mode().to(dtype=weight_dtype)
                elif getattr(args, "is_preprocessed_dataset", False):
                    # For preprocessed dataset, we have params (concatenated mean + logvar)
                    # batch["instance_latents_params"] shape is (B, 2*C, H, W)
                    
                    # Data is already on device if we moved it in Dataset
                    inst_params = batch["instance_latents_params"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                    cond_params = batch["cond_latents_params"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
                    
                    # We can use the mean (first half of channels) as the mode
                    channels = inst_params.shape[1] // 2
                    model_input = inst_params[:, :channels, :, :]
                    cond_model_input = cond_params[:, :channels, :, :]
                else:
                    with offload_models(vae, device=accelerator.device, offload=args.offload):
                        pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                        cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.mode()
                    cond_model_input = vae.encode(cond_pixel_values).latent_dist.mode()
                    # Cast latents to weight_dtype immediately after VAE encoding to match transformer dtype
                    model_input = model_input.to(dtype=weight_dtype)
                    cond_model_input = cond_model_input.to(dtype=weight_dtype)

                model_input = Flux2KleinPipeline._patchify_latents(model_input)
                model_input = (model_input - latents_bn_mean) / latents_bn_std

                cond_model_input = Flux2KleinPipeline._patchify_latents(cond_model_input)
                cond_model_input = (cond_model_input - latents_bn_mean) / latents_bn_std

                model_input_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(device=model_input.device)
                cond_model_input_ids = Flux2KleinPipeline._prepare_latent_ids(cond_model_input).to(
                    device=cond_model_input.device
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # [B, C, H, W] -> [B, H*W, C]
                packed_noisy_model_input = Flux2KleinPipeline._pack_latents(noisy_model_input)
                packed_cond_model_input = Flux2KleinPipeline._pack_latents(cond_model_input)

                orig_input_shape = packed_noisy_model_input.shape
                orig_input_ids_shape = model_input_ids.shape

                # concatenate the model inputs with the cond inputs
                packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
                model_input_ids = torch.cat([model_input_ids, cond_model_input_ids], dim=1)

                # handle guidance
                guidance = torch.full([1], args.guidance_scale, device=accelerator.device)
                guidance = guidance.expand(model_input.shape[0])

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input.to(dtype=weight_dtype),  # (B, image_seq_len, C)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds_batch.to(dtype=weight_dtype),
                    txt_ids=text_ids_batch,  # B, text_seq_len, 4
                    img_ids=model_input_ids,  # B, image_seq_len, 4
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : orig_input_shape[1], :]
                model_input_ids = model_input_ids[:, : orig_input_ids_shape[1], :]

                model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, model_input_ids)

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Gradient clipping
                    # For fp16: unscale gradients first, then clip, then scaler handles the rest
                    # For bf16/fp32: no scaler, clip directly
                    if accelerator.scaler is not None:
                        # fp16 with scaler: unscale first, then clip
                        accelerator.unscale_gradients(optimizer)
                        params_to_clip = [p for p in transformer.parameters() if p.requires_grad and p.grad is not None]
                        if len(params_to_clip) > 0:
                            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    else:
                        # No scaler (bf16 or fp32) - safe to clip directly
                        params_to_clip = [p for p in transformer.parameters() if p.requires_grad and p.grad is not None]
                        if len(params_to_clip) > 0:
                            torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                # Optimizer step - works for fp16 (with scaler), bf16, and fp32
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or is_fsdp:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )

                del pipeline
                free_memory()

    # Save the lora layers
    accelerator.wait_for_everyone()

    if is_fsdp:
        transformer = unwrap_model(transformer)
        state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        modules_to_save = {}
        if is_fsdp:
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    state_dict = {
                        k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }
                else:
                    state_dict = {
                        k: v.to(weight_dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()
                    }

            transformer_lora_layers = get_peft_model_state_dict(
                transformer,
                state_dict=state_dict,
            )
            transformer_lora_layers = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in transformer_lora_layers.items()
            }

        else:
            transformer = unwrap_model(transformer)
            if args.bnb_quantization_config_path is None:
                if args.upcast_before_saving:
                    transformer.to(torch.float32)
                else:
                    transformer = transformer.to(weight_dtype)
            transformer_lora_layers = get_peft_model_state_dict(transformer)

        modules_to_save["transformer"] = transformer

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        images = []
        run_validation = (args.validation_prompt and args.num_validation_images > 0) or (args.final_validation_prompt)
        should_run_final_inference = not args.skip_final_inference and run_validation
        if should_run_final_inference:
            pipeline = Flux2KleinPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            # load attention processors
            pipeline.load_lora_weights(args.output_dir)

            # run inference
            images = []
            if args.validation_prompt and args.num_validation_images > 0:
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=validation_kwargs,
                    epoch=epoch,
                    is_final_validation=True,
                    torch_dtype=weight_dtype,
                )
            del pipeline
            free_memory()

        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt
        save_model_card(
            (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
            images=images,
            base_model=args.pretrained_model_name_or_path,
            instance_prompt=args.validation_prompt or "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination.",
            validation_prompt=validation_prompt,
            repo_folder=args.output_dir,
            fp8_training=args.do_fp8_training,
        )

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

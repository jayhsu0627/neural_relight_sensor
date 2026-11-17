#!/bin/bash
# Delight LoRA exactly as in the paper: FLUX.2-klein-4b, rank 16, alpha 4, lr 1e-4, fp8,
# 10 epochs x 128 pairs = 1,280 steps, batch size 1, 1024 px (about 40 min on one RTX PRO 6000).
# Writes ./runs/delight_lora/pytorch_lora_weights.safetensors (+ checkpoints every 200 steps).
# Extra arguments are passed through, e.g. --validation_image_dir path/to/dark/images
set -e
cd "$(dirname "$0")/.."
accelerate launch train/train_delight_lora_flux2_klein.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4b \
  --base_model_name_or_path black-forest-labs/FLUX.2-klein-4b \
  --dataset_name linoyts/flux-delight-dataset \
  --image_column output_image \
  --cond_image_column input_image \
  --caption_column prompt \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 10 \
  --learning_rate 1e-4 \
  --rank 16 \
  --lora_alpha 4 \
  --output_dir runs/delight_lora \
  --checkpointing_steps 200 \
  --validation_prompt "Relight the image to remove all existing lighting conditions and replace them with neutral, uniform illumination." \
  --validation_epochs 50 \
  --mixed_precision bf16 \
  --do_fp8_training \
  --report_to wandb \
  "$@"

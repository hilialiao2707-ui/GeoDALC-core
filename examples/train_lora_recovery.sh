#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to a compressed .pt checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora_attn_r4}"

python scripts/build_lora_data.py \
  --src data/geogpt_qa.jsonl \
  --out_dir data/lora_splits \
  --seed 42

python train_lora_recovery.py \
  --base_model "${BASE_MODEL}" \
  --train_file data/lora_splits/lora_train_instruction.jsonl \
  --val_file data/lora_splits/lora_val_instruction.jsonl \
  --output_dir "${OUTPUT_DIR}" \
  --max_length 512 \
  --epochs 2 \
  --lr 2e-4 \
  --batch_size 1 \
  --grad_accum 16 \
  --lora_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --target_modules q_u_proj,q_v_proj,k_u_proj,k_v_proj,v_u_proj,v_v_proj,o_u_proj,o_v_proj

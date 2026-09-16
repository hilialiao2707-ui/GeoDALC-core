#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to a compressed .pt checkpoint}"
ADAPTER_PATH="${ADAPTER_PATH:?Set ADAPTER_PATH to the saved LoRA adapter directory}"

python eval_lora_recovery.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --dataset wikitext2 \
  --seq_len 512 \
  --batch_size 1 \
  --output_json outputs/wikitext_ppl.json

python eval_lora_recovery.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --dataset geogpt \
  --seq_len 512 \
  --batch_size 1 \
  --output_json outputs/geogpt_ppl.json

python eval_geogpt_qa_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --data_path data/geogpt_qa.jsonl \
  --split test \
  --output_path outputs/geogpt_qa.json

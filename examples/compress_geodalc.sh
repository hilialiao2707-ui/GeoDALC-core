#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/compression}"
RATIO="${RATIO:-0.2}"
CALIBRATION="${CALIBRATION:-wikitext2_geogpt20}"

mkdir -p "${OUTPUT_DIR}"

python compress_geodalc.py \
  --model "${MODEL}" \
  --step 1 \
  --ratio "${RATIO}" \
  --dataset "${CALIBRATION}" \
  --whitening_nsamples 256 \
  --seed 42 \
  --model_seq_len 2048 \
  --save_path "${OUTPUT_DIR}" \
  --run_low_resource

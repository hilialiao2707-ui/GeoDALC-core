# GeoDALC core code

This repository snapshot contains the two core parts of GeoDALC:

1. domain-aware, activation-guided low-rank compression;
2. LoRA recovery applied to the factorized projection matrices.

Paper-writing utilities, figures, experiment logs, remote orchestration scripts, checkpoints, and datasets are intentionally excluded.

## Repository layout

```text
compress_geodalc.py          # compression entry point
component/                   # factorized LLaMA/Mistral/OPT layers
utils/data_utils.py          # WikiText/geoscience mixed calibration
utils/model_utils.py         # model and checkpoint loading
utils/peft/                  # experiment-compatible PEFT snapshot
train_lora_recovery.py       # post-compression LoRA training
eval_lora_recovery.py        # perplexity evaluation
eval_geogpt_qa_lora.py       # GeoGPT-QA token-F1 evaluation
scripts/build_lora_data.py   # deterministic LoRA split builder
examples/                    # minimal shell commands
```

The reported study uses LLaMA-2-7B. Other model components remain only because the compression entry point imports them directly; they were not the primary evaluated configuration.

## Installation

Python 3.9 or 3.10 and a CUDA-enabled PyTorch environment are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Access to gated LLaMA checkpoints must be configured separately through Hugging Face.

## Data

Place the geoscience QA JSONL at `data/geogpt_qa.jsonl`. See [`data/README.md`](data/README.md) for the accepted schema and environment-variable overrides. The dataset itself is ignored by Git.

Calibration names accepted by the core loader are:

- `wikitext2`
- `geogpt`
- `wikitext2_geogpt5`
- `wikitext2_geogpt10`
- `wikitext2_geogpt15`
- `wikitext2_geogpt20`
- `wikitext2_geogpt30`

`wikitext2_geogpt20` is the main mixed-calibration setting used in the paper. The suffix follows the experiment code: the loader adds geoscience records equal to 20% of the WikiText record count before shuffling.

## 1. GeoDALC compression

The `--ratio` argument is the paper's compression ratio. Internally the factorized layers retain `1 - ratio` of the original parameter budget.

```bash
bash examples/compress_geodalc.sh
```

Equivalent explicit command:

```bash
python compress_geodalc.py \
  --model meta-llama/Llama-2-7b-hf \
  --step 1 \
  --ratio 0.2 \
  --dataset wikitext2_geogpt20 \
  --whitening_nsamples 256 \
  --seed 42 \
  --model_seq_len 2048 \
  --save_path outputs/compression \
  --run_low_resource
```

The command writes a profiling matrix and a serialized compressed checkpoint. These files are excluded by `.gitignore`.

## 2. LoRA recovery

Build deterministic train, validation, and held-out splits:

```bash
python scripts/build_lora_data.py \
  --src data/geogpt_qa.jsonl \
  --out_dir data/lora_splits \
  --seed 42
```

Train the main attention-only rank-4 adapter:

```bash
BASE_MODEL=/path/to/compressed_model.pt \
bash examples/train_lora_recovery.sh
```

The adapter targets both factors of each compressed attention projection:

```text
q_u_proj,q_v_proj,k_u_proj,k_v_proj,
v_u_proj,v_v_proj,o_u_proj,o_v_proj
```

To reproduce LoRA-All, append the factorized MLP projections:

```text
gate_u_proj,gate_v_proj,down_u_proj,down_v_proj,up_u_proj,up_v_proj
```

## Evaluation

```bash
BASE_MODEL=/path/to/compressed_model.pt \
ADAPTER_PATH=/path/to/adapter \
bash examples/evaluate_lora.sh
```

The evaluation scripts report WikiText perplexity, geoscience-text perplexity, and GeoGPT-QA exact match/token F1.

## Reproducibility notes

- Calibration and LoRA splits are deterministic for the documented seeds.
- Checkpoints are stored with `torch.save` and contain Python model objects; load them only from trusted sources.
- The repository does not redistribute LLaMA weights or the geoscience dataset.
- This snapshot preserves the experiment-compatible PEFT implementation under `utils/peft`.

## Acknowledgement and license

The compression implementation builds on [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM). See [`NOTICE.md`](NOTICE.md) for third-party notices. The code is distributed under the Apache License 2.0.

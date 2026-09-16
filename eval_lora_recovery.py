import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.model_utils import get_model_from_local  # noqa: E402
from utils.peft import PeftModel  # noqa: E402
from evaluater import ppl_eval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate compressed model + LoRA recovery adapter.")
    parser.add_argument("--base_model", type=str, required=True, help="Compressed .pt checkpoint path.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Saved LoRA adapter directory.")
    parser.add_argument("--dataset", type=str, choices=["wikitext2", "geogpt"], required=True)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = get_model_from_local(args.base_model)
    model = model.float().to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    result = ppl_eval(
        model,
        tokenizer,
        datasets=[args.dataset],
        model_seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = {
        "base_model": args.base_model,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "result": result,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

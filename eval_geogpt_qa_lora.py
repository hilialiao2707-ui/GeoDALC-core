import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.model_utils import get_model_from_local  # noqa: E402
from utils.peft import PeftModel  # noqa: E402


DEFAULT_GEOGPT_JSONL = ROOT / "data" / "geogpt_qa.jsonl"


def load_geogpt_records(
    data_path=DEFAULT_GEOGPT_JSONL,
    split="test",
    max_samples=None,
    split_seed=42,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("GeoGPT-QA split ratios must sum to 1.0")
    rows = []
    with Path(data_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    random.Random(split_seed).shuffle(rows)
    total = len(rows)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    splits = {
        "train": rows[:train_end],
        "validation": rows[train_end:val_end],
        "test": rows[val_end:],
    }
    out = splits[split]
    if max_samples is not None and max_samples > 0:
        out = out[:max_samples]
    return out


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    common = 0
    for token, count in pred_counts.items():
        common += min(count, ref_counts.get(token, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, device: str, max_new_tokens: int = 128):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = output[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text.split("\n")[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate compressed model + LoRA adapter on GeoGPT-QA.")
    parser.add_argument("--base_model", required=True, help="Compressed .pt checkpoint path.")
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter directory.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_GEOGPT_JSONL), help="GeoGPT-QA JSONL file.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = get_model_from_local(args.base_model)
    model = model.eval().float().to(args.device)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    qa_rows = load_geogpt_records(
        args.data_path,
        args.split,
        max_samples=args.max_samples,
        split_seed=42,
    )
    rows = []
    exact_match = 0
    total_f1 = 0.0

    for idx, row in enumerate(qa_rows):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        prediction = generate_answer(model, tokenizer, question, args.device, max_new_tokens=args.max_new_tokens)
        exact = int(normalize_text(prediction) == normalize_text(answer))
        f1 = token_f1(prediction, answer)
        exact_match += exact
        total_f1 += f1
        rows.append(
            {
                "id": int(row.get("index", idx)),
                "question": question,
                "reference_answer": answer,
                "prediction": prediction,
                "exact_match": bool(exact),
                "token_f1": f1,
            }
        )

    count = max(len(rows), 1)
    summary = {
        "model_name": Path(args.base_model).name,
        "adapter_path": args.adapter_path,
        "split": args.split,
        "num_questions": len(rows),
        "exact_match": exact_match / count,
        "token_f1": total_f1 / count,
        "results": rows,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("model_name", "adapter_path", "exact_match", "token_f1", "num_questions")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

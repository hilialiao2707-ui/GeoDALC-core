import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_INSTRUCTION = "Answer the following geoscience question."


def parse_args():
    parser = argparse.ArgumentParser(description="Build LoRA train/val splits from GeoGPT-QA train jsonl.")
    parser.add_argument(
        "--src",
        type=str,
        default="data/geogpt_qa.jsonl",
        help="Source GeoGPT-QA train jsonl path.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/lora_splits",
        help="Output directory for train/val jsonl files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    return parser.parse_args()


def build_text(question: str, answer: str) -> str:
    return (
        "### Instruction:\n"
        f"{DEFAULT_INSTRUCTION}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer:\n"
        f"{answer}"
    )


def normalize_record(obj: dict) -> Optional[dict]:
    source_index = obj.get("index")
    question = (obj.get("question") or obj.get("input") or obj.get("query") or "").strip()
    answer = (obj.get("answer") or obj.get("output") or obj.get("response") or "").strip()
    if not question or not answer:
        text = (obj.get("text") or "").strip()
        if not text:
            return None
        return {
            "source_index": source_index,
            "instruction": DEFAULT_INSTRUCTION,
            "input": "",
            "output": text,
            "text": text,
        }
    return {
        "source_index": source_index,
        "instruction": DEFAULT_INSTRUCTION,
        "input": question,
        "output": answer,
        "text": build_text(question, answer),
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            row = normalize_record(obj)
            if row is not None:
                rows.append(row)

    random.Random(args.seed).shuffle(rows)
    n_train = int(len(rows) * args.train_ratio)
    n_val = int(len(rows) * args.val_ratio)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]

    write_jsonl(out_dir / "lora_train_instruction.jsonl", train_rows)
    write_jsonl(out_dir / "lora_val_instruction.jsonl", val_rows)
    write_jsonl(out_dir / "heldout_test_instruction.jsonl", test_rows)

    meta = {
        "source": str(src),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "num_total": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "num_test": len(test_rows),
        "train_file": str(out_dir / "lora_train_instruction.jsonl"),
        "val_file": str(out_dir / "lora_val_instruction.jsonl"),
        "test_file": str(out_dir / "heldout_test_instruction.jsonl"),
    }
    with (out_dir / "lora_split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

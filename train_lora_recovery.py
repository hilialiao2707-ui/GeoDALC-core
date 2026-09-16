import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset
import transformers


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.model_utils import get_model_from_local  # noqa: E402
from utils.peft import LoraConfig, get_peft_model, get_peft_model_state_dict  # noqa: E402


ANSWER_MARKER = "### Answer:\n"


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA recovery fine-tuning on compressed LLaMA checkpoints.")
    parser.add_argument("--base_model", type=str, required=True, help="Compressed .pt checkpoint path.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_u_proj,q_v_proj,k_u_proj,k_v_proj,v_u_proj,v_v_proj,o_u_proj,o_v_proj",
        help="Comma-separated factorized projection names. The default is attention-only GeoDALC recovery.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


class JsonlInstructionDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if not text:
                    instruction = (obj.get("instruction") or "Answer the following geoscience question.").strip()
                    question = (obj.get("input") or obj.get("question") or "").strip()
                    answer = (obj.get("output") or obj.get("answer") or "").strip()
                    if question or answer:
                        text = (
                            "### Instruction:\n"
                            f"{instruction}\n\n"
                            "### Question:\n"
                            f"{question}\n\n"
                            "### Answer:\n"
                            f"{answer}"
                        ).strip()
                if text:
                    self.samples.append(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        labels = enc["input_ids"].copy()

        if ANSWER_MARKER in text:
            prefix = text.split(ANSWER_MARKER, 1)[0] + ANSWER_MARKER
            prefix_ids = self.tokenizer(
                prefix,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
                add_special_tokens=True,
            )["input_ids"]
            prefix_len = min(len(prefix_ids), len(labels))
            labels[:prefix_len] = [-100] * prefix_len

        enc["labels"] = labels
        return enc


def build_model(base_model: str, target_modules: List[str], lora_r: int, lora_alpha: int, lora_dropout: float):
    model, tokenizer = get_model_from_local(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    tokenizer.padding_side = "right"

    model = model.half()
    model.config.use_cache = False

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model, tokenizer


def save_lora_adapter(model, tokenizer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    state_dict = get_peft_model_state_dict(model)
    torch.save(state_dict, os.path.join(output_dir, "adapter_model.bin"))
    model.peft_config["default"].save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    transformers.set_seed(args.seed)
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    model, tokenizer = build_model(
        args.base_model,
        target_modules,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
    )

    train_dataset = JsonlInstructionDataset(args.train_file, tokenizer, args.max_length)
    val_dataset = JsonlInstructionDataset(args.val_file, tokenizer, args.max_length)

    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, int(len(train_loader) * args.epochs / args.grad_accum))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and not args.bf16)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    global_step = 0

    def move_batch(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def run_eval():
        model.eval()
        losses = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                batch = move_batch(batch)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                    outputs = model(**batch)
                losses.append(outputs.loss.detach().float().item())
                if step >= 63:
                    break
        model.train()
        return sum(losses) / max(1, len(losses))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(args.epochs)):
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "step": global_step,
                                "train_loss": float(loss.detach().float().item() * args.grad_accum),
                                "lr": float(scheduler.get_last_lr()[0]),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

                if global_step % args.eval_steps == 0:
                    val_loss = run_eval()
                    print(json.dumps({"step": global_step, "val_loss": float(val_loss)}, ensure_ascii=False), flush=True)

    save_lora_adapter(model, tokenizer, args.output_dir)
    print(f"LoRA recovery adapter saved to: {args.output_dir}")

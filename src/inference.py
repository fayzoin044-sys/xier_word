import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainConfig


def read_system_prompt(train_file: Path) -> str:
    with train_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            return row["prompt"][0]["content"]
    raise ValueError(f"训练文件为空：{train_file}")


def load_model(model_source: str, adapter_dir: Path, config: TrainConfig):
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"找不到 LoRA adapter：{adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        dtype=config.compute_dtype,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(config.device).eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, config: TrainConfig, user_message: str, max_new_tokens: int):
    messages = [
        {"role": "system", "content": read_system_prompt(config.train_file)},
        {"role": "user", "content": user_message},
    ]
    batch = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(config.device)
    output = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output[0, batch["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    config = TrainConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter", type=Path, default=config.adapter_dir
    )
    parser.add_argument("--prompt", required=True, help="当前客户消息")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load_model(
        config.model_name_or_path, args.adapter, config
    )
    print(generate(model, tokenizer, config, args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()

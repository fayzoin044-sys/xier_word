import argparse
import math
import random
from contextlib import nullcontext

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from config import TrainConfig
from dataset import get_dataloader


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="GLM-4-9B LoRA 训练")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="只训练 32 条、验证 16 条，并保存到 checkpoints_v2_smoke",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(config: TrainConfig):
    if config.device.type == "cuda":
        return torch.autocast("cuda", dtype=config.compute_dtype)
    return nullcontext()


@torch.no_grad()
def evaluate(model, dev_loader: DataLoader, config: TrainConfig) -> float:
    model.eval()
    losses = []

    for batch in tqdm(dev_loader, desc="evaluate", leave=False):
        batch = {name: tensor.to(config.device)
                 for name, tensor in batch.items()}
        with autocast_context(config):
            loss = model(**batch).loss
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main() -> None:

    args = parse_args()
    config = TrainConfig()
    if args.smoke:
        config.max_train_samples = 32
        config.max_dev_samples = 16
        config.output_dir = (
            config.output_dir.parent / f"{config.output_dir.name}_smoke"
        )
        print(
            "冒烟模式：训练 32 条、验证 16 条，"
            f"结果保存到 {config.output_dir}"
        )

    set_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    if config.device.type != "cuda":
        raise RuntimeError("GLM-4-9B 训练需要 CUDA 显卡")

    print(f"设备：{config.device}，计算精度：{config.compute_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path, use_fast=True)

    train_loader = get_dataloader(
        file_name=config.train_file,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        max_samples=config.max_train_samples,
        shuffle=True,
    )
    dev_loader = get_dataloader(
        file_name=config.dev_file,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
        max_samples=config.max_dev_samples,
        shuffle=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=config.compute_dtype,
        low_cpu_mem_usage=True,

    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.to(config.device)
    model.print_trainable_parameters()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.compute_dtype == torch.float16
    )

    best_dev_loss = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}")

        for micro_step, batch in enumerate(progress, start=1):
            batch = {name: tensor.to(config.device)
                     for name, tensor in batch.items()}

            with autocast_context(config):
                loss = model(**batch).loss

                loss_for_backward = loss / config.gradient_accumulation_steps

            scaler.scale(loss_for_backward).backward()

            should_update = (
                micro_step % config.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )

            if not should_update:
                continue

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.max_grad_norm
            )

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.logging_steps == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        dev_loss = evaluate(model, dev_loader, config)
        print(f"epoch={epoch} dev_loss={dev_loss:.4f}")

        model.save_pretrained(config.output_dir / "last")
        tokenizer.save_pretrained(config.output_dir / "last")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            model.save_pretrained(config.output_dir / "best")
            tokenizer.save_pretrained(config.output_dir / "best")
            print("已保存新的最佳 LoRA adapter")

    print(f"训练完成，最佳 dev_loss={best_dev_loss:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_utils import TextClassificationDataset, build_loader, class_weights, load_labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device: torch.device, criterion) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate", leave=False):
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits
            losses.append(criterion(logits, labels).item())
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": metrics.accuracy_score(targets, predictions),
        "macro_f1": metrics.f1_score(
            targets, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": metrics.f1_score(
            targets, predictions, average="weighted", zero_division=0
        ),
        "targets": targets,
        "predictions": predictions,
    }


def run_training(args, use_chat_template: bool) -> None:
    set_seed(args.seed)
    labels = load_labels(args.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_loader = build_loader(
        args.data_dir / "train.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        True,
        use_chat_template,
    )
    dev_loader = build_loader(
        args.data_dir / "dev.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        False,
        use_chat_template,
    )
    test_loader = build_loader(
        args.data_dir / "test.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        False,
        use_chat_template,
    )
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    weights = class_weights(train_loader.dataset, len(labels)).to(
        device=device, dtype=next(model.parameters()).dtype
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            labels_tensor = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits
            loss = criterion(logits, labels_tensor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        dev_metrics = evaluate(model, dev_loader, device, criterion)
        print(
            f"epoch={epoch} train_loss={sum(losses) / len(losses):.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_accuracy={dev_metrics['accuracy']:.4f} "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )
        if float(dev_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(dev_metrics["macro_f1"])
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir
    ).to(device)
    test_metrics = evaluate(best_model, test_loader, device, criterion)
    report = metrics.classification_report(
        test_metrics["targets"],
        test_metrics["predictions"],
        target_names=labels,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    result = {
        "accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "classification_report": report,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

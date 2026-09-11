import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn import metrics
from transformers import BertTokenizer

from bert1 import AiModel
from data_utils import build_loader, load_labels


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="BERT 动态 INT8 量化与评估")
    parser.add_argument("--base-model", default="google-bert/bert-base-chinese")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--labels", type=Path, default=project_dir / "data" / "labels.txt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_dir / "models" / "bert-finetuned" / "classification_best1.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "models" / "bert-int8" / "classification_quantized.pt",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    loader = build_loader(
        args.data_dir / "test.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        False,
        False,
    )
    model = AiModel(args.base_model, len(labels))
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model, args.output)
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch_labels = batch.pop("labels")
            logits = quantized_model(
                batch["input_ids"],
                batch.get("token_type_ids"),
                batch["attention_mask"],
            )
            predictions.extend(torch.argmax(logits, dim=-1).tolist())
            targets.extend(batch_labels.tolist())
    print(f"accuracy={metrics.accuracy_score(targets, predictions):.4f}")
    print(
        metrics.classification_report(
            targets,
            predictions,
            target_names=labels,
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()

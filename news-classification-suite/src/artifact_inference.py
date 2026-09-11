import argparse
from pathlib import Path

import torch
from transformers import BertTokenizer

from bert1 import AiModel
from data_utils import load_labels
from textcnn_model import StudentTextCNN


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("bert", "bert-int8", "textcnn"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--base-model", default="google-bert/bert-base-chinese")
    parser.add_argument("--labels", type=Path, default=project_dir / "data" / "labels.txt")
    parser.add_argument("--max-length", type=int, default=32)
    return parser.parse_args()


def load_model(args: argparse.Namespace, tokenizer: BertTokenizer, num_labels: int):
    if args.model == "bert":
        model = AiModel(args.base_model, num_labels)
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model
    if args.model == "textcnn":
        model = StudentTextCNN(tokenizer.vocab_size, num_labels=num_labels)
        state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model
    return torch.load(args.weights, map_location="cpu", weights_only=False)


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    encoded = tokenizer(
        args.text,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    model = load_model(args, tokenizer, len(labels))
    model.eval()
    with torch.no_grad():
        if args.model == "textcnn":
            logits = model(encoded["input_ids"])
        else:
            logits = model(
                encoded["input_ids"],
                encoded.get("token_type_ids"),
                encoded["attention_mask"],
            )
    index = int(torch.argmax(logits, dim=-1).item())
    print(labels[index])


if __name__ == "__main__":
    main()

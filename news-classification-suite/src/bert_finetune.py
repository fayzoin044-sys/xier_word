import argparse
from pathlib import Path

from transformer_training import run_training


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="BERT 新闻分类全量微调")
    parser.add_argument("--model", default="google-bert/bert-base-chinese")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--labels", type=Path,
                        default=project_dir / "data" / "labels.txt")
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "outputs" / "bert"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args(), use_chat_template=False)

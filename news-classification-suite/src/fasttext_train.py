import argparse
import tempfile
from pathlib import Path

import fasttext

from data_utils import TextClassificationDataset, load_labels


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="FastText 新闻分类")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--labels", type=Path,
                        default=project_dir / "data" / "labels.txt")
    parser.add_argument(
        "--output", type=Path, default=project_dir / "outputs" / "fasttext.bin"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--word-ngrams", type=int, default=2)
    return parser.parse_args()


def write_fasttext(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for text, label in rows:
            normalized = " ".join(text.replace("\t", " ").split())
            segmented = " ".join(
                character for character in normalized if character != " ")
            stream.write(f"__label__{label} {segmented}\n")


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    train_data = TextClassificationDataset(
        args.data_dir / "train.txt", len(labels))
    test_data = TextClassificationDataset(
        args.data_dir / "test.txt", len(labels))
    with tempfile.TemporaryDirectory() as temp_dir:
        train_path = Path(temp_dir) / "train.txt"
        test_path = Path(temp_dir) / "test.txt"
        write_fasttext(train_path, train_data.rows)
        write_fasttext(test_path, test_data.rows)
        model = fasttext.train_supervised(
            input=str(train_path),
            lr=args.learning_rate,
            epoch=args.epochs,
            wordNgrams=args.word_ngrams,
            loss="softmax",
        )
        sample_count, precision, recall = model.test(str(test_path))
        print(
            f"samples={sample_count} precision={precision:.4f} recall={recall:.4f}"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.output))


if __name__ == "__main__":
    main()

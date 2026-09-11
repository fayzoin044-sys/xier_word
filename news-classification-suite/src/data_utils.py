from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TextClassificationDataset(Dataset):
    def __init__(self, path: Path, label_count: int) -> None:
        self.rows: list[tuple[str, int]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = line.rstrip("\n")
                if not value:
                    continue
                try:
                    text, raw_label = value.rsplit("\t", 1)
                    label = int(raw_label)
                except ValueError as error:
                    raise ValueError(
                        f"{path.name}:{line_number} 格式无效") from error
                if not 0 <= label < label_count:
                    raise ValueError(f"{path.name}:{line_number} 标签越界")
                self.rows.append((text, label))
        if not self.rows:
            raise ValueError(f"数据集为空：{path.name}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.rows[index]


def load_labels(path: Path) -> list[str]:
    labels = [line.strip()
              for line in path.read_text(encoding="utf-8").splitlines()]
    labels = [label for label in labels if label]
    if len(labels) != len(set(labels)):
        raise ValueError("类别名称存在重复")
    return labels


class TransformerCollator:
    def __init__(self, tokenizer, max_length: int, use_chat_template: bool) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template

    def __call__(self, rows: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts = [text for text, _ in rows]
        labels = torch.tensor([label for _, label in rows], dtype=torch.long)
        if self.use_chat_template:
            conversations = [
                [{"role": "user", "content": text}]
                for text in texts
            ]
            encoded = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        encoded["labels"] = labels
        return encoded


def build_loader(
    path: Path,
    labels: list[str],
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    use_chat_template: bool,
) -> DataLoader:
    dataset = TextClassificationDataset(path, len(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TransformerCollator(
            tokenizer, max_length, use_chat_template),
    )


def class_weights(dataset: TextClassificationDataset, label_count: int) -> torch.Tensor:
    labels = torch.tensor(
        [label for _, label in dataset.rows], dtype=torch.long)
    counts = torch.bincount(labels, minlength=label_count).float()
    if torch.any(counts == 0):
        raise ValueError(f"训练集存在空类别：{counts.tolist()}")
    return len(labels) / (label_count * counts)

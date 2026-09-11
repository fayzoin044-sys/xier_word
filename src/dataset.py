from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


def apply_template(
    tokenizer: Any,
    messages: list[dict],
    generation_prompt: bool,
) -> list[int]:

    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    if hasattr(result, "keys"):
        result = result["input_ids"]
    return list(result)


def encode_sample(
    sample: dict,
    tokenizer: Any,
    max_length: int,
) -> dict[str, list[int]]:

    prompt = sample["prompt"]
    completion = sample["completion"]

    prompt_ids = apply_template(
        tokenizer,
        prompt,
        generation_prompt=True,
    )

    input_ids = apply_template(
        tokenizer,
        prompt + completion,
        generation_prompt=False,
    )
    if input_ids[-1] != tokenizer.eos_token_id:
        input_ids.append(tokenizer.eos_token_id)

    while len(input_ids) > max_length and len(prompt) > 2:
        prompt = prompt[:1] + prompt[2:]
        prompt_ids = apply_template(
            tokenizer,
            prompt,
            generation_prompt=True,
        )
        input_ids = apply_template(
            tokenizer,
            prompt + completion,
            generation_prompt=False,
        )
        if input_ids[-1] != tokenizer.eos_token_id:
            input_ids.append(tokenizer.eos_token_id)

    if len(input_ids) > max_length:
        raise ValueError(f"存在超过 max_length={max_length} 的样本")
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template 无法正确区分 prompt 和 completion")

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class ChatCollator:

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:

        encoded_samples = [
            encode_sample(sample, self.tokenizer, self.max_length)
            for sample in samples
        ]
        max_length = max(len(sample["input_ids"])
                         for sample in encoded_samples)

        def pad(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_length - len(values))

        return {
            "input_ids": torch.tensor(
                [
                    pad(sample["input_ids"], self.pad_token_id)
                    for sample in encoded_samples
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [pad(sample["attention_mask"], 0)
                 for sample in encoded_samples],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [pad(sample["labels"], -100) for sample in encoded_samples],
                dtype=torch.long,
            ),
        }


def get_dataloader(
    file_name: Path,
    tokenizer: Any,
    batch_size: int,
    max_length: int,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> DataLoader:

    dataset = load_dataset(
        "json",
        data_files=str(file_name),
        split="train",
    )

    if max_samples is not None:
        sample_count = min(max_samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    collate_fn = ChatCollator(tokenizer, max_length)

    my_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return my_dataloader

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:

    model_name_or_path: str = os.environ.get(
        "BASE_MODEL_PATH", str(PROJECT_DIR / "model_glm")
    )

    train_file: Path = Path(
        os.environ.get("TRAIN_FILE", str(PROJECT_DIR / "data" / "train.jsonl"))
    )

    dev_file: Path = Path(
        os.environ.get("DEV_FILE", str(PROJECT_DIR / "data" / "dev.jsonl"))
    )

    output_dir: Path = PROJECT_DIR / "outputs"
    adapter_dir: Path = PROJECT_DIR / "artifacts" / "best"

    max_length: int = 2048

    batch_size: int = 1

    gradient_accumulation_steps: int = 16

    epochs: int = 1

    learning_rate: float = 2e-4

    weight_decay: float = 0.01

    warmup_ratio: float = 0.03

    max_grad_norm: float = 1.0

    logging_steps: int = 10

    seed: int = 42

    max_train_samples: int | None = None
    max_dev_samples: int | None = 1000

    lora_rank: int = 16

    lora_alpha: int = 32

    lora_dropout: float = 0.05

    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    @property
    def device(self) -> torch.device:

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def compute_dtype(self) -> torch.dtype:

        if self.device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

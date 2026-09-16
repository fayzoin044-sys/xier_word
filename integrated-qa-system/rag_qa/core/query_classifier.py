"""基于本地中文 BERT 的查询二分类器。

本模块使用手写 PyTorch 训练循环完成 BERT 微调，不依赖 Hugging Face
``Trainer``。分类目标固定为：

- ``通用知识``：不需要查询项目知识库，可直接交给大模型回答；
- ``专业咨询``：需要进入后续 RAG 检索流程。

训练数据和基础模型均从当前项目读取。训练产生的最佳权重会保存到独立目录，
不会覆盖原始的 ``bert-base-chinese``。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


# 保证直接运行本文件时也能导入项目根目录下的 base 包。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base.logger import logger


RAG_QA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = (
    RAG_QA_DIR / "classify_data" / "model_generic_5000.json"
)
DEFAULT_BASE_MODEL_DIR = RAG_QA_DIR / "models" / "bert-base-chinese"
DEFAULT_TRAINED_MODEL_DIR = RAG_QA_DIR / "models" / "bert_query_classifier"

LABEL_TO_ID = {
    "通用知识": 0,
    "专业咨询": 1,
}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}


def set_random_seed(seed: int) -> None:
    """固定随机种子，使抽样、数据切分和训练尽量可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QueryDataset(Dataset):
    """保存尚未分词的查询文本和数字标签。"""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class QueryClassifierModel(nn.Module):
    """沿用手写方式：BERT 主体后连接一个二分类线性层。"""

    def __init__(self, base_model_path: str | Path):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(base_model_path))
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = self.dropout(bert_output.pooler_output)
        return self.classifier(pooled_output)


class QueryClassifier:
    """负责训练、保存、加载和调用查询二分类模型。"""

    def __init__(
        self,
        base_model_path: str | Path = DEFAULT_BASE_MODEL_DIR,
        trained_model_path: str | Path = DEFAULT_TRAINED_MODEL_DIR,
        max_length: int = 128,
    ):
        self.base_model_path = Path(base_model_path).resolve()
        self.trained_model_path = Path(trained_model_path).resolve()
        self.max_length = max_length

        # 当前项目明确使用 CPU，不启用 CUDA、FP16 或梯度检查点。
        self.device = torch.device("cpu")
        self.model: QueryClassifierModel | None = None

        if not self.base_model_path.exists():
            raise FileNotFoundError(
                f"BERT 基础模型目录不存在：{self.base_model_path}"
            )

        # 训练完成后优先使用保存到微调目录中的 tokenizer；首次训练使用基础模型。
        tokenizer_path = (
            self.trained_model_path
            if (self.trained_model_path / "tokenizer_config.json").exists()
            else self.base_model_path
        )
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        logger.info("QueryClassifier 初始化完成，运行设备：CPU")

    @property
    def weights_path(self) -> Path:
        """微调后最佳模型权重的保存位置。"""

        return self.trained_model_path / "classifier_state.pt"

    def _load_records(
        self,
        data_file: str | Path,
    ) -> list[dict[str, Any]]:
        """读取 JSON Lines 数据、校验标签并按 query 去重。"""

        data_path = Path(data_file).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"分类训练数据不存在：{data_path}")

        unique_records: dict[str, dict[str, Any]] = {}
        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"训练数据第 {line_number} 行不是合法 JSON"
                    ) from exc

                query = str(item.get("query", "")).strip()
                label = str(item.get("label", "")).strip()

                if not query:
                    raise ValueError(f"训练数据第 {line_number} 行 query 为空")
                if label not in LABEL_TO_ID:
                    raise ValueError(
                        f"训练数据第 {line_number} 行标签无效：{label!r}"
                    )

                previous = unique_records.get(query)
                if previous is not None:
                    if previous["label_name"] != label:
                        raise ValueError(
                            f"相同问题存在冲突标签：{query!r}"
                        )
                    continue

                unique_records[query] = {
                    "query": query,
                    "label": LABEL_TO_ID[label],
                    "label_name": label,
                }

        records = list(unique_records.values())
        if len(records) < 2:
            raise ValueError("有效分类数据少于2条，无法划分训练集和验证集")

        logger.info(
            "分类数据读取完成：原始文件 %s，去重后 %d 条",
            data_path,
            len(records),
        )
        return records

    def _prepare_datasets(
        self,
        data_file: str | Path,
        max_samples: int | None,
        validation_ratio: float,
        random_seed: int,
    ) -> tuple[QueryDataset, QueryDataset]:
        """分层抽样并按照标签比例切分训练集和验证集。"""

        if not 0 < validation_ratio < 1:
            raise ValueError("validation_ratio 必须在 0 和 1 之间")

        records = self._load_records(data_file)

        if max_samples is not None:
            if max_samples < 2:
                raise ValueError("max_samples 不能小于2")

            if len(records) > max_samples:
                labels = [record["label"] for record in records]
                records, _ = train_test_split(
                    records,
                    train_size=max_samples,
                    stratify=labels,
                    random_state=random_seed,
                )

        labels = [record["label"] for record in records]
        train_records, validation_records = train_test_split(
            records,
            test_size=validation_ratio,
            stratify=labels,
            random_state=random_seed,
        )

        train_distribution = Counter(
            record["label_name"] for record in train_records
        )
        validation_distribution = Counter(
            record["label_name"] for record in validation_records
        )
        logger.info(
            "数据切分完成：训练集 %d 条 %s，验证集 %d 条 %s",
            len(train_records),
            dict(train_distribution),
            len(validation_records),
            dict(validation_distribution),
        )

        return QueryDataset(train_records), QueryDataset(validation_records)

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """将一批查询文本统一分词并转换为 PyTorch 张量。"""

        queries = [item["query"] for item in batch]
        labels = torch.tensor(
            [item["label"] for item in batch],
            dtype=torch.long,
        )
        encoded = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        encoded["labels"] = labels
        return encoded

    def _create_dataloader(
        self,
        dataset: QueryDataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """创建适合当前 Windows CPU 环境的数据加载器。"""

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

    def _evaluate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> dict[str, Any]:
        """在验证集上计算损失、准确率和分类指标。"""

        if self.model is None:
            raise RuntimeError("模型尚未初始化")

        self.model.eval()
        total_loss = 0.0
        predictions: list[int] = []
        true_labels: list[int] = []

        with torch.no_grad():
            for batch in dataloader:
                labels = batch.pop("labels").to(self.device)
                model_inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }
                logits = self.model(**model_inputs)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                predictions.extend(
                    torch.argmax(logits, dim=-1).cpu().tolist()
                )
                true_labels.extend(labels.cpu().tolist())

        average_loss = total_loss / max(len(dataloader), 1)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        report = classification_report(
            true_labels,
            predictions,
            labels=[0, 1],
            target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
            digits=4,
            zero_division=0,
        )
        matrix = confusion_matrix(
            true_labels,
            predictions,
            labels=[0, 1],
        )

        return {
            "loss": average_loss,
            "accuracy": accuracy,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": matrix,
        }

    def _save_best_model(
        self,
        training_metadata: dict[str, Any],
    ) -> None:
        """保存最佳权重、分词器和训练参数，不覆盖基础模型。"""

        if self.model is None:
            raise RuntimeError("模型尚未初始化")

        self.trained_model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        self.tokenizer.save_pretrained(str(self.trained_model_path))

        metadata_path = self.trained_model_path / "training_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                training_metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

        logger.info("当前最佳分类模型已保存：%s", self.weights_path)

    def train_model(
        self,
        data_file: str | Path = DEFAULT_DATA_FILE,
        max_samples: int | None = 500,
        validation_ratio: float = 0.2,
        batch_size: int = 8,
        epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """使用手写 PyTorch 循环在 CPU 上微调查询分类模型。

        默认从数据文件中分层抽取500条，再按80%/20%切分训练集和验证集。
        每轮结束进行一次验证，并按照最低验证损失保存最佳模型。
        """

        if batch_size <= 0 or epochs <= 0:
            raise ValueError("batch_size 和 epochs 必须大于0")
        if learning_rate <= 0:
            raise ValueError("learning_rate 必须大于0")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio 必须在 [0, 1) 范围内")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm 必须大于0")

        set_random_seed(random_seed)
        train_dataset, validation_dataset = self._prepare_datasets(
            data_file=data_file,
            max_samples=max_samples,
            validation_ratio=validation_ratio,
            random_seed=random_seed,
        )
        train_dataloader = self._create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        validation_dataloader = self._create_dataloader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        self.model = QueryClassifierModel(self.base_model_path).to(self.device)
        criterion = nn.CrossEntropyLoss(reduction="mean")
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_training_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_training_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        logger.info(
            "开始微调查询分类模型：CPU，训练集=%d，验证集=%d，"
            "epoch=%d，batch_size=%d，总步数=%d，warmup步数=%d",
            len(train_dataset),
            len(validation_dataset),
            epochs,
            batch_size,
            total_training_steps,
            warmup_steps,
        )

        best_validation_loss = float("inf")
        best_epoch = 0
        best_metrics: dict[str, Any] = {}

        for epoch_index in range(epochs):
            self.model.train()
            running_loss = 0.0
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch_index + 1}/{epochs}",
            )

            for step, batch in enumerate(progress_bar, start=1):
                labels = batch.pop("labels").to(self.device)
                model_inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(**model_inputs)
                loss = criterion(logits, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=max_grad_norm,
                )
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                if logging_steps > 0 and step % logging_steps == 0:
                    logger.info(
                        "Epoch %d/%d，step %d/%d，当前loss=%.4f，学习率=%.8f",
                        epoch_index + 1,
                        epochs,
                        step,
                        len(train_dataloader),
                        loss.item(),
                        scheduler.get_last_lr()[0],
                    )

            average_training_loss = running_loss / max(
                len(train_dataloader),
                1,
            )
            validation_metrics = self._evaluate(
                validation_dataloader,
                criterion,
            )

            logger.info(
                "Epoch %d/%d完成：train_loss=%.4f，val_loss=%.4f，"
                "accuracy=%.4f，precision=%.4f，recall=%.4f，f1=%.4f",
                epoch_index + 1,
                epochs,
                average_training_loss,
                validation_metrics["loss"],
                validation_metrics["accuracy"],
                validation_metrics["precision"],
                validation_metrics["recall"],
                validation_metrics["f1"],
            )
            logger.info(
                "验证集分类报告：\n%s",
                validation_metrics["classification_report"],
            )
            logger.info(
                "验证集混淆矩阵：\n%s",
                validation_metrics["confusion_matrix"],
            )

            if validation_metrics["loss"] < best_validation_loss:
                best_validation_loss = validation_metrics["loss"]
                best_epoch = epoch_index + 1
                best_metrics = {
                    "epoch": best_epoch,
                    "train_loss": average_training_loss,
                    "validation_loss": validation_metrics["loss"],
                    "accuracy": validation_metrics["accuracy"],
                    "precision": validation_metrics["precision"],
                    "recall": validation_metrics["recall"],
                    "f1": validation_metrics["f1"],
                }
                self._save_best_model(
                    {
                        "labels": LABEL_TO_ID,
                        "base_model_path": str(self.base_model_path),
                        "data_file": str(Path(data_file).resolve()),
                        "max_samples": max_samples,
                        "validation_ratio": validation_ratio,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "warmup_ratio": warmup_ratio,
                        "warmup_steps": warmup_steps,
                        "max_grad_norm": max_grad_norm,
                        "max_length": self.max_length,
                        "random_seed": random_seed,
                        "best_metrics": best_metrics,
                    }
                )

        logger.info(
            "查询分类模型训练完成，最佳轮次=%d，最佳验证损失=%.4f",
            best_epoch,
            best_validation_loss,
        )
        return best_metrics

    def load_trained_model(self) -> None:
        """加载已经保存的最佳微调权重，用于后续预测。"""

        if not self.weights_path.exists():
            raise FileNotFoundError(
                "尚未找到微调后的查询分类模型，请先运行训练："
                f"{self.weights_path}"
            )

        model = QueryClassifierModel(self.base_model_path).to(self.device)
        try:
            state_dict = torch.load(
                self.weights_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            # 兼容尚不支持 weights_only 参数的旧版 PyTorch。
            state_dict = torch.load(
                self.weights_path,
                map_location=self.device,
            )

        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        logger.info("已加载微调后的查询分类模型：%s", self.weights_path)

    def predict_category(self, query: str) -> str:
        """预测单个用户问题，返回“通用知识”或“专业咨询”。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("待分类的 query 必须是非空字符串")

        if self.model is None:
            self.load_trained_model()

        if self.model is None:
            raise RuntimeError("微调模型加载失败")

        encoded = self.tokenizer(
            query.strip(),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**model_inputs)
            predicted_id = int(torch.argmax(logits, dim=-1).item())

        category = ID_TO_LABEL[predicted_id]
        logger.info("查询分类完成：%s -> %s", query, category)
        return category


def main() -> None:
    """提供手动训练和单条预测入口，不在导入模块时自动训练。"""

    parser = argparse.ArgumentParser(description="BERT 查询二分类器")
    parser.add_argument(
        "--mode",
        choices=("train", "predict"),
        required=True,
        help="train=训练并验证；predict=加载最佳模型预测单条问题",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="JSON Lines 分类数据文件",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="分层抽取的数据量；默认500条",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="predict 模式下要分类的用户问题",
    )
    args = parser.parse_args()

    classifier = QueryClassifier()

    if args.mode == "train":
        metrics = classifier.train_model(
            data_file=args.data_file,
            max_samples=args.max_samples,
        )
        print("训练完成，最佳验证结果：")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if not args.query:
        parser.error("predict 模式必须提供 --query")
    print("分类结果：", classifier.predict_category(args.query))


if __name__ == "__main__":
# #训练时
#     classifier = QueryClassifier()
#     classifier.train_model()
# 训练结束进行评估和预测
    classifier = QueryClassifier()

    query = input("请输入要测试的问题：")
    result = classifier.predict_category(query)

    print("分类结果：", result)

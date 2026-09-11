import argparse
from pathlib import Path

import torch
import torch.nn.functional as functional
from sklearn import metrics
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer

from bert1 import AiModel
from data_utils import build_loader, load_labels
from textcnn_model import StudentTextCNN


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="BERT 到 TextCNN 的离线知识蒸馏")
    parser.add_argument("--base-model", default="google-bert/bert-base-chinese")
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--labels", type=Path, default=project_dir / "data" / "labels.txt")
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=project_dir / "models" / "bert-finetuned" / "classification_best1.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "models" / "textcnn-distilled" / "student_textcnn_best.bin",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--eval-steps", type=int, default=100)
    return parser.parse_args()


def record_teacher_knowledge(model, loader, device: torch.device):
    model.eval()
    knowledge = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="record teacher logits"):
            labels = batch.pop("labels")
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(
                inputs["input_ids"],
                inputs.get("token_type_ids"),
                inputs["attention_mask"],
            )
            knowledge.append(
                (inputs["input_ids"].cpu(), labels.cpu(), logits.cpu())
            )
    return knowledge


def distillation_loss(
    student_logits,
    labels,
    teacher_logits,
    alpha: float,
    temperature: float,
):
    soft_student = functional.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = functional.softmax(teacher_logits / temperature, dim=1)
    soft_loss = functional.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean",
    )
    hard_loss = functional.cross_entropy(student_logits, labels)
    return (
        alpha * temperature * temperature * soft_loss
        + (1.0 - alpha) * hard_loss
    )


def evaluate(model, loader, device: torch.device):
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            logits = model(batch["input_ids"].to(device))
            losses.append(functional.cross_entropy(logits, labels).item())
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return sum(losses) / len(losses), predictions, targets


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    train_loader = build_loader(
        args.data_dir / "train.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        True,
        False,
    )
    dev_loader = build_loader(
        args.data_dir / "dev.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        False,
        False,
    )
    test_loader = build_loader(
        args.data_dir / "test.txt",
        labels,
        tokenizer,
        args.batch_size,
        args.max_length,
        False,
        False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = AiModel(args.base_model, len(labels)).to(device)
    teacher.load_state_dict(
        torch.load(args.teacher_weights, map_location=device, weights_only=True)
    )
    knowledge = record_teacher_knowledge(teacher, train_loader, device)
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    student = StudentTextCNN(tokenizer.vocab_size, num_labels=len(labels)).to(device)
    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_dev_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        student.train()
        for step, (input_ids, batch_labels, teacher_logits) in enumerate(
            tqdm(knowledge, desc=f"epoch {epoch}/{args.epochs}"),
            start=1,
        ):
            student_logits = student(input_ids.to(device))
            loss = distillation_loss(
                student_logits,
                batch_labels.to(device),
                teacher_logits.to(device),
                args.alpha,
                args.temperature,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % args.eval_steps == 0 or step == len(knowledge):
                dev_loss, predictions, targets = evaluate(student, dev_loader, device)
                accuracy = metrics.accuracy_score(targets, predictions)
                print(
                    f"epoch={epoch} step={step} train_loss={loss.item():.4f} "
                    f"dev_loss={dev_loss:.4f} dev_accuracy={accuracy:.4f}"
                )
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(student.state_dict(), args.output)
                student.train()
    student.load_state_dict(
        torch.load(args.output, map_location=device, weights_only=True)
    )
    _, predictions, targets = evaluate(student, test_loader, device)
    print(f"test_accuracy={metrics.accuracy_score(targets, predictions):.4f}")
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

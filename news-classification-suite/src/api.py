import os

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_dir = os.environ.get("MODEL_DIR", "outputs/qwen")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_dir).to(device).eval()
app = Flask(__name__)


@app.post("/classify")
def classify():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    if str(getattr(model.config, "model_type", "")).startswith("qwen"):
        encoded = tokenizer.apply_chat_template(
            [[{"role": "user", "content": text}]],
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=128,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
    else:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    label_id = int(torch.argmax(logits, dim=-1).item())
    label = model.config.id2label.get(label_id, str(label_id))
    return jsonify({"label_id": label_id, "label": label})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5004")))

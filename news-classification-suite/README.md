# 中文新闻多分类与模型轻量化

本项目围绕 10 类中文新闻分类，对比传统机器学习、FastText、BERT 和 Qwen2.5 全量微调，并实践 BERT 动态量化与 BERT → TextCNN 知识蒸馏。仓库包含统一数据集、训练与推理源码、三个本地模型产物，以及独立发布的 Qwen 全量微调模型。

## 项目结果

| 方案 | 测试准确率 | 模型产物 |
| --- | ---: | --- |
| TF-IDF + 随机森林 | 约 81% | 不发布权重，可由源码复现 |
| FastText | 约 91.5% | 不发布权重，可由源码复现 |
| BERT 全量微调 | 约 94.6% | Git LFS |
| Qwen2.5-1.5B 全量微调 | 接近 94% | Hugging Face |
| BERT → TextCNN 蒸馏 | 约 88.7% | Git LFS |
| BERT INT8 动态量化 | 轻量化实验 | Git LFS |

其中 Qwen 使用约十分之一训练数据取得接近 BERT 的分类效果；TextCNN 蒸馏模型约 11.6 MB，相比 409 MB 的 BERT state dict 大幅缩小。

## 类别

`finance`、`realty`、`stocks`、`education`、`science`、`society`、`politics`、`sports`、`game`、`entertainment`

## 项目结构

```text
data/                     25,000 条训练、5,000 条验证、5,000 条测试数据
models/bert-finetuned/    BERT 全量微调权重
models/bert-int8/         BERT INT8 动态量化模型
models/textcnn-distilled/ TextCNN 蒸馏权重
src/random_forest_train.py
src/fasttext_train.py
src/bert_finetune.py
src/qwen_full_finetune.py
src/artifact_inference.py 已发布模型的统一推理入口
src/api.py                Flask 推理接口
```

## 数据格式

每行由新闻文本、制表符和数字标签组成：

```text
新闻文本<TAB>0
```

标签范围为 0 到 9，顺序与 `data/labels.txt` 一致。

## 安装

```bash
git lfs pull
python -m venv .venv
python -m pip install -r requirements.txt
```

## 训练

```bash
python src/random_forest_train.py
python src/fasttext_train.py
python src/bert_finetune.py --model google-bert/bert-base-chinese
python src/qwen_full_finetune.py --model Qwen/Qwen2.5-1.5B-Instruct --gradient-checkpointing
```

Qwen 脚本更新模型全部参数，属于全量微调，不使用 LoRA 或其他 PEFT 方法。

## 已保存模型推理

```bash
python src/artifact_inference.py --model bert \
  --weights models/bert-finetuned/classification_best1.bin \
  --text "中国队在本届比赛中取得胜利"

python src/artifact_inference.py --model bert-int8 \
  --weights models/bert-int8/classification_quantized.pt \
  --text "中国队在本届比赛中取得胜利"

python src/artifact_inference.py --model textcnn \
  --weights models/textcnn-distilled/student_textcnn_best.bin \
  --text "中国队在本届比赛中取得胜利"
```

各模型的格式和使用说明见 [`models/`](models/)。Qwen 全量微调模型已发布至 [Hugging Face](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier)。

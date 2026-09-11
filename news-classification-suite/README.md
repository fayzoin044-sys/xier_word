# 多模型中文新闻分类

本项目整理自一套 10 类中文新闻分类实验，统一包含随机森林、FastText、BERT 和 Qwen 全量微调实现，并提供 Flask 推理接口。

## 类别

`finance`、`realty`、`stocks`、`education`、`science`、`society`、`politics`、`sports`、`game`、`entertainment`

## 项目结构

```text
data/labels.txt              类别名称
src/data_utils.py            数据读取与批处理
src/random_forest_train.py   TF-IDF + 随机森林
src/fasttext_train.py        FastText 训练与评估
src/bert_finetune.py         BERT 全量微调
src/qwen_full_finetune.py    Qwen 全量微调
src/transformer_training.py  Transformer 公共训练流程
src/api.py                   Flask 推理接口
```

## 数据格式

仓库不包含原始新闻文本。请准备 `data/train.txt`、`data/dev.txt` 和 `data/test.txt`，每行由文本、制表符和数字标签组成：

```text
新闻文本<TAB>0
```

标签范围为 0 到 9，顺序与 `data/labels.txt` 一致。

## 安装

```bash
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

Qwen 脚本会更新模型全部参数，因此属于全量微调，不使用 LoRA 或其他 PEFT 方法。模型输出默认保存到 `outputs/`，该目录不会提交到 Git。

## API

```bash
MODEL_DIR=outputs/qwen python src/api.py
```

请求示例：

```bash
curl -X POST http://127.0.0.1:5004/classify -H "Content-Type: application/json" -d '{"text":"示例新闻文本"}'
```

公开仓库只包含整理后的源码与类别表，不包含原始数据、训练权重、IDE 配置、缓存、日志、内部路径或个人笔记。

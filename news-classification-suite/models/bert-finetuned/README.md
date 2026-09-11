# BERT 全量微调模型

`classification_best1.bin` 是基于中文 BERT 完成 10 类新闻分类全量微调后保存的 state dict，包含 BERT 主体和线性分类头参数，文件大小约 409 MB。

模型结构为 `BertModel` 加 `Linear(768, 10)`，训练时没有冻结 BERT 参数。使用 `src/artifact_inference.py` 加载时，需要能够访问兼容的 `google-bert/bert-base-chinese` 底座配置与词表。

```bash
python src/artifact_inference.py \
  --model bert \
  --weights models/bert-finetuned/classification_best1.bin \
  --text "中国队在本届比赛中取得胜利"
```

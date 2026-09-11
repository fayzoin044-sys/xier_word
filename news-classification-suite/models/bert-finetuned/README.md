# BERT 全量微调模型

`classification_best1.bin` 是基于中文 BERT 完成 10 类新闻分类全量微调后保存的 state dict，包含 BERT 主体和线性分类头参数，文件大小约 409 MB。

模型结构为 `BertModel` 加 `Linear(768, 10)`，训练时没有冻结 BERT 参数。使用 `src/artifact_inference.py` 加载时，需要能够访问兼容的 `google-bert/bert-base-chinese` 底座配置与词表。

对应训练实现为 [`src/bert_finetune.py`](../../src/bert_finetune.py)，公开版保留原实验的 3 轮训练、batch size 128、最大长度 32、学习率 `5e-5` 和按验证损失保存最佳模型的设置。

```bash
python src/artifact_inference.py \
  --model bert \
  --weights models/bert-finetuned/classification_best1.bin \
  --text "中国队在本届比赛中取得胜利"
```

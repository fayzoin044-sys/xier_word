# BERT INT8 动态量化模型

`classification_quantized.pt` 由全量微调 BERT 分类器进行 PyTorch Linear 层动态 INT8 量化得到，文件大小约 153 MB，仅用于 CPU 推理。

该文件是历史实验保存的 PyTorch 完整模型对象。为了兼容其序列化类路径，仓库保留了精简的 `src/bert1.py` 模型定义。

对应生成与评估实现为 [`src/bert_quantize.py`](../../src/bert_quantize.py)，加载 BERT 微调 state dict 后，对全部 `Linear` 层进行动态 INT8 量化。

```bash
python src/artifact_inference.py \
  --model bert-int8 \
  --weights models/bert-int8/classification_quantized.pt \
  --text "中国队在本届比赛中取得胜利"
```

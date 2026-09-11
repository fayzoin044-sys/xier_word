# TextCNN 知识蒸馏模型

`student_textcnn_best.bin` 是以全量微调 BERT 为教师模型、TextCNN 为学生模型得到的 state dict，文件大小约 11.6 MB。

学生网络包含 128 维词嵌入、卷积核尺寸为 3/4/5 的三路卷积、最大池化、Dropout 和 10 类分类头。蒸馏损失结合真实标签交叉熵与教师输出 KL 散度，温度为 2.0，蒸馏权重为 0.8。

对应训练实现为 [`src/bert_textcnn_distill.py`](../../src/bert_textcnn_distill.py)，先离线记录 BERT 教师 logits，再释放教师模型并训练学生网络。

```bash
python src/artifact_inference.py \
  --model textcnn \
  --weights models/textcnn-distilled/student_textcnn_best.bin \
  --text "中国队在本届比赛中取得胜利"
```

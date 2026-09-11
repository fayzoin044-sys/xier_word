# 模型产物

本目录保存三个可直接关联到项目源码的本地模型产物。Qwen2.5 全量微调模型体积约 3.09 GB，单独发布到 [Hugging Face](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier)。

| 目录 | 模型 | 大小 | 格式 |
| --- | --- | ---: | --- |
| `bert-finetuned/` | BERT 全量微调分类器 | 409 MB | PyTorch state dict |
| `bert-int8/` | BERT Linear 层动态 INT8 量化模型 | 153 MB | PyTorch 完整模型 |
| `textcnn-distilled/` | BERT 教师蒸馏后的 TextCNN | 11.6 MB | PyTorch state dict |

三个文件均使用 Git LFS 管理。克隆仓库后先运行 `git lfs pull`。

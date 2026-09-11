# 中文 NLP 模型项目合集

本仓库整理了两个完整的中文 NLP 项目，覆盖大语言模型 LoRA 微调、文本多分类、知识蒸馏、INT8 量化和在线推理。源码、整理后的数据集、评估结果及可公开的最终模型产物均已提供。

## 项目概览

| 项目 | 核心工作 | 关键结果 | 项目与模型 |
| --- | --- | --- | --- |
| GLM-4-9B 多轮客服对话模型微调 | 12 类客服策略识别与结构化回复生成；清洗 16 万余条多轮对话，构建 3 万条类别均衡训练数据；使用 LoRA 完成多轮 SFT | 策略准确率 `42.33% → 69.50%`；Macro-F1 `0.4151 → 0.6837`；JSON Schema 合规率 `100%` | [项目说明](glm-lora-customer-service/README.md) · [LoRA Adapter](glm-lora-customer-service/artifacts/best/) · [测试结果](glm-lora-customer-service/results/) |
| 中文文本多分类与模型轻量化 | 对比随机森林、FastText、BERT 与 Qwen2.5-1.5B；完成 BERT 全量微调、TextCNN 知识蒸馏、INT8 动态量化和 Flask 推理接口 | BERT `94.6%`；Qwen 接近 `94%`；蒸馏 TextCNN `88.7%`，模型体积约缩小 15 倍 | [项目说明](news-classification-suite/README.md) · [BERT/量化/蒸馏模型](news-classification-suite/models/) · [Qwen 模型](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier) |

## 快速入口

- [GLM 项目：训练、推理、评估和模型使用方法](glm-lora-customer-service/README.md)
- [中文新闻分类项目：各模型方案、轻量化流程和部署方法](news-classification-suite/README.md)
- [Hugging Face：Qwen2.5-1.5B 中文新闻分类全量微调模型](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier)

公开仓库仅排除了个人笔记、内部注释、IDE/缓存文件、运行日志及本机路径等与项目复现无关的内容。

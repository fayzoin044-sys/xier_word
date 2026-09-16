# AI 模型训练与应用项目合集

本仓库整理了三个模型训练与应用项目，覆盖 SDXL 图像风格 LoRA 微调、大语言模型 LoRA 微调、文本多分类、知识蒸馏、INT8 量化和在线推理。源码、整理后的数据集、评估结果及可公开的最终模型产物均已提供。

## 项目概览

| 项目 | 核心工作 | 关键结果 | 项目与模型 |
| --- | --- | --- | --- |
| GLM-4-9B 多轮客服对话模型微调 | 12 类客服策略识别与结构化回复生成；清洗 16 万余条多轮对话，构建 3 万条类别均衡训练数据；使用 LoRA 完成多轮 SFT | 策略准确率 `42.33% → 69.50%`；Macro-F1 `0.4151 → 0.6837`；JSON Schema 合规率 `100%` | [项目说明](glm-lora-customer-service/README.md) · [LoRA Adapter](glm-lora-customer-service/artifacts/best/) · [测试结果](glm-lora-customer-service/results/) |
| 中文文本多分类与模型轻量化 | 对比随机森林、FastText、BERT 与 Qwen2.5-1.5B；完成 BERT 全量微调、TextCNN 知识蒸馏、INT8 动态量化和 Flask 推理接口 | BERT `94.6%`；Qwen 接近 `94%`；蒸馏 TextCNN `88.7%`，模型体积约缩小 15 倍 | [项目说明](news-classification-suite/README.md) · [BERT/量化/蒸馏模型](news-classification-suite/models/) · [Qwen 模型](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier) |
| SDXL 梵高风格 LoRA 图像微调 | 画作去重与数据划分、UNet LoRA 训练、照片图生图、同种子基础模型对比 | 358 张训练画作；2000 步；RTX 5090 训练约 74 分钟；24 张开发集对比 | [项目与效果展示](vangogh-lora/README.md) · [LoRA 权重](vangogh-lora/outputs/vangogh-sdxl-v1/) |

## 快速入口

- [SDXL 梵高风格项目：效果对比、训练记录和图生图使用方法](vangogh-lora/README.md)
- [GLM 项目：训练、推理、评估和模型使用方法](glm-lora-customer-service/README.md)
- [中文新闻分类项目：各模型方案、轻量化流程和部署方法](news-classification-suite/README.md)
- [Hugging Face：Qwen2.5-1.5B 中文新闻分类全量微调模型](https://huggingface.co/zyhForHugging/qwen2.5-1.5b-chinese-news-classifier)

公开仓库排除了个人笔记、IDE/缓存、重复训练检查点和优化器状态。图像项目保留原始实验记录中的服务器路径用于追溯，实际运行请按项目说明配置路径。

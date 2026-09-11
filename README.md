# 中文 NLP 模型项目合集

本仓库集中整理了两个中文自然语言处理项目：根目录为 GLM-4-9B LoRA 客服策略微调项目，`news-classification-suite/` 为多模型中文新闻分类项目。公开版本已移除原始数据、内部注释、项目笔记、运行日志、本地路径和敏感信息。

## 项目导航

- 当前根目录：[GLM-4-9B LoRA 客服策略微调](#glm-4-9b-lora-客服策略微调)
- 子目录：[多模型中文新闻分类](news-classification-suite/README.md)

## GLM-4-9B LoRA 客服策略微调

这是一个经过公开发布清理的 GLM-4-9B LoRA 微调项目，包含训练、推理、评估、数据清洗源码和最佳适配器。原始对话数据、逐条评估记录、运行日志、内部路径及项目笔记未包含在仓库中。

## 项目结构

```text
artifacts/best/        最佳 LoRA 适配器与 tokenizer
results/metrics.json  600 条样本的汇总评估指标
src/config.py         训练配置
src/dataset.py        数据编码与批处理
src/train.py          LoRA 训练入口
src/inference.py      单条推理入口
src/evaluate.py       基础模型与 LoRA 对比评估
tools/                数据划分、审查与清洗工具
requirements.txt      Python 依赖
```

## 环境安装

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

训练和评估需要能够加载 GLM-4-9B 的 CUDA 环境。基础模型默认从项目根目录的 `model_glm` 读取，也可以通过 `BASE_MODEL_PATH` 指定本地目录或模型标识。

## 数据格式

训练集与验证集使用 JSONL，每行结构如下：

```json
{"prompt":[{"role":"system","content":"任务说明"},{"role":"user","content":"用户消息"}],"completion":[{"role":"assistant","content":"{\"strategy\":\"策略类别\",\"response\":\"回复内容\"}"}],"target_strategy":"策略类别"}
```

默认路径为 `data/train.jsonl` 和 `data/dev.jsonl`。仓库不包含原始数据集。

## 训练

```bash
python src/train.py --smoke
python src/train.py
```

训练输出保存在 `outputs/`。可使用环境变量覆盖路径：

```bash
BASE_MODEL_PATH=/path/to/model TRAIN_FILE=/path/to/train.jsonl DEV_FILE=/path/to/dev.jsonl python src/train.py
```

## 推理

```bash
python src/inference.py --adapter artifacts/best --prompt "请输入测试消息"
```

## 评估

评估脚本通过环境变量读取评审服务密钥，不在源码或配置文件中保存密钥。

```bash
DEEPSEEK_API_KEY=your_key python src/evaluate.py --adapter artifacts/best --dev-file data/dev.jsonl --samples-per-class 50
```

## 评估结果

| 指标 | 基础模型 | LoRA 模型 |
| --- | ---: | ---: |
| JSON 有效率 | 91.67% | 100.00% |
| Schema 有效率 | 91.50% | 100.00% |
| 策略准确率 | 42.33% | 69.50% |
| Macro F1 | 0.4151 | 0.6837 |
| 平均损失 | 2.0957 | 1.0313 |
| 回答评分 | 3.2067 | 3.5117 |

适配器权重和 tokenizer 通过 Git LFS 管理。克隆后运行 `git lfs pull` 获取完整文件。

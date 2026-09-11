# GLM-4-9B 多轮客服对话 LoRA 微调

面向中文多轮客服场景的策略识别与结构化回复生成项目。项目完成了会话级数据划分、语义清洗、类别均衡、LoRA 微调以及基础模型对照评估，并公开最终 Adapter、可复现实验数据和评估明细。

## 项目概览

| 项目 | 说明 |
| --- | --- |
| 任务 | 识别 12 类客服策略，并生成符合 JSON Schema 的客服回复 |
| 基础模型 | GLM-4-9B |
| 微调方式 | SFT + LoRA，注入 Q/K/V/O 投影层 |
| 训练参数 | 约 1,606 万，占模型参数约 0.17% |
| 训练环境 | 单张 32GB GPU，BF16 |
| 公开数据 | 30,000 条训练、1,000 条验证、4,380 条测试数据 |
| 模型产物 | 最佳 LoRA Adapter 与 tokenizer，使用 Git LFS 管理 |

## 核心结果

测试集从 12 个类别各抽取 50 条，共 600 条，与未微调基础模型进行对照。

| 指标 | 基础模型 | LoRA 模型 |
| --- | ---: | ---: |
| 策略准确率 | 42.33% | **69.50%** |
| Macro-F1 | 0.4151 | **0.6837** |
| JSON 有效率 | 91.67% | **100.00%** |
| JSON Schema 合规率 | 91.50% | **100.00%** |
| 平均损失 | 2.0957 | **1.0313** |
| LLM-as-a-Judge 回复评分 | 3.2067 | **3.5117** |

## 处理流程

```mermaid
flowchart LR
    A[16 万余条多轮数据] --> B[按完整会话划分]
    B --> C[规则审查与语义清洗]
    C --> D[12 类均衡采样 3 万条]
    D --> E[Chat Template 多轮 SFT]
    E --> F[GLM-4-9B + LoRA]
    F --> G[600 条对照评估]
```

## 关键实现

- 按完整会话划分训练、验证和测试集，三个集合的会话重叠数均为 0。
- 历史对话作为上下文输入，仅对当前 Assistant 的策略和回复计算损失。
- 超长会话优先裁剪早期历史，保留最近上下文及完整目标回复。
- 训练集按 12 类各采样 2,500 条，缓解长尾类别不均衡。
- 评估同时覆盖分类指标、JSON 结构合规性、生成损失和 LLM-as-a-Judge 评分。

## 目录结构

```text
data/                  训练、验证和测试数据
artifacts/best/        最佳 LoRA Adapter 与 tokenizer
results/               汇总指标和 600 条逐条评估结果
reports/               数据集与清洗统计
src/                   训练、推理与评估源码
tools/                 数据审查、清洗、重划分和采样工具
requirements.txt       Python 依赖
```

## 快速开始

```bash
python -m venv .venv
python -m pip install -r requirements.txt
git lfs pull
```

准备兼容的 GLM-4-9B 基础模型后，可通过环境变量指定路径：

```bash
BASE_MODEL_PATH=/path/to/model python src/inference.py \
  --adapter artifacts/best \
  --prompt "请输入测试消息"
```

重新训练与评估：

```bash
BASE_MODEL_PATH=/path/to/model python src/train.py
DEEPSEEK_API_KEY=your_key BASE_MODEL_PATH=/path/to/model \
  python src/evaluate.py --adapter artifacts/best --dev-file data/test.jsonl --samples-per-class 50
```

数据格式和划分方法见 [data/README.md](data/README.md)，实验详情见 [results/README.md](results/README.md)。

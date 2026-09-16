# RAG 离线评测

本目录用于在不开虚拟机、不启动 MySQL/Redis/Milvus 的情况下，评估当前项目的 RAG 检索与重排链路。

## 保留与替换的部分

- 真实保留：`LLM基础知识.pdf` 解析、父子分块、BGE-M3 dense/sparse 向量、`WeightedRanker(1.0, 0.7)` 的 IP 归一化融合、父块去重、BGE Reranker、Top-2。
- 离线替换：Milvus 的 IVF_FLAT/倒排索引查询改为 35 个子块上的精确内积。
- 默认禁用：DeepSeek 生成与 Ragas 的 LLM-as-a-judge 指标，避免 API 成本。

因此，这套结果可以说明检索相关性和重排效果，不能用来说明 Milvus 的 ANN 召回损失、并发能力或线上延迟。

## 测试集

`data/rag_retrieval_golden.jsonl` 当前包含：

- 24 条有答案问题：直接问法、改写、事实题和跨块推理题；
- 5 条 hard negative：答案不在 PDF 中；
- 每条正例包含人工标注的父块序号和参考答案。

父块序号由真实分块顺序解析为当前运行的 `parent_id`，这样项目移动目录后稳定 ID 发生变化也能继续评测。每次运行还会导出完整 `parent_manifest_*.json` 供人工复核。

## 运行

在项目根目录执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python evaluation/rag/evaluate_rag_offline.py
```

首次运行会生成文档向量缓存，后续会复用；修改 PDF 或分块参数后缓存自动失效。强制重建可添加 `--rebuild-cache`，快速跳过重排模型可添加 `--skip-reranker`。

## 指标

- `Hit@1`：第一个父块是否命中任一金标准父块；
- `Hit@K/M`：返回列表是否命中任一金标准父块；
- `MRR`：第一个相关父块排名倒数；
- `ID Precision / Recall`：按父块 ID 计算的宏平均精确率和召回率；
- hard negative 不相关上下文暴露率：当前检索没有拒答阈值，知识库非空就会返回上下文，因此该值会如实暴露而不会包装成“拒答成功率”。

下一阶段如果要评估最终回答，再在相同测试集上补充 Ragas 的 Faithfulness、Answer Relevancy、Context Precision/Recall；这一步需要生成答案和 LLM judge，默认不自动产生费用。

## GPT + Ragas 生成质量评测

`evaluate_ragas_gpt.py` 读取已经保存的 Hybrid + Reranker Top-2 结果，不会重新连接数据库或加载本地向量模型。它使用 GPT 生成回答，并用 Ragas 0.4 collections API 计算：

- Context Precision；
- Context Recall；
- Faithfulness；
- Answer Relevancy；
- Answer Correctness。

此外，脚本会把 hard negative 单独汇总，并用固定拒答文本计算正确拒答率，不与知识库内问题混合平均。

安装依赖：

```powershell
python -m pip install -r evaluation/rag/requirements-ragas.txt
```

先执行不产生费用的 dry-run：

```powershell
python evaluation/rag/evaluate_ragas_gpt.py
```

不要把 API Key 写进文件或粘贴到聊天。可在当前 PowerShell 会话中遮罩输入，然后先跑默认3条冒烟测试：

```powershell
$env:OPENAI_API_KEY = Read-Host 'OpenAI API Key' -MaskInput
python evaluation/rag/evaluate_ragas_gpt.py --execute
```

确认模型权限和结果正常后，再运行全部29条：

```powershell
python evaluation/rag/evaluate_ragas_gpt.py --execute --limit 0
Remove-Item Env:OPENAI_API_KEY
```

默认使用 `gpt-5.6-luna` 生成和评审、`text-embedding-3-small` 计算 Answer Relevancy/Correctness。模型可以通过 `--generation-model`、`--judge-model` 和 `--embedding-model` 覆盖。

脚本有三层费用保护：默认 dry-run、默认只选3条、生成及Ragas内部请求写入本地缓存。重复运行相同模型、问题和上下文时会尽量复用缓存。API Key 只从对应的环境变量（`OPENAI_API_KEY` 或 `OPENROUTER_API_KEY`）读取，不会被保存进缓存或报告。

### 使用 OpenRouter

如果 Key 以 `sk-or-v1-` 开头，它属于 OpenRouter，不能直接作为官方 OpenAI Key 使用。先把它放进 `OPENROUTER_API_KEY`，然后指定提供商：

```powershell
$env:OPENROUTER_API_KEY = Read-Host 'OpenRouter API Key' -MaskInput
python evaluation/rag/evaluate_ragas_gpt.py --api-provider openrouter --execute
```

OpenRouter 模式使用 OpenAI-compatible Chat Completions，默认模型为 `openai/gpt-4o-mini`；Answer Relevancy 和 Answer Correctness 所需的向量由项目本地 BGE-M3 计算，不调用或计费外部 Embedding API。全量运行添加 `--limit 0`。

如果不熟悉 PowerShell 环境变量，可以直接运行一键脚本。它会隐藏读取新 Key、运行默认3条测试，并在结束后自动清除当前进程中的 Key：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\evaluation\rag\run_openrouter_ragas.ps1
```

3条冒烟测试通过后，全量29条使用：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\evaluation\rag\run_openrouter_ragas.ps1 -Full
```

### Ragas 0.4.3 兼容说明

Ragas 0.4.3 会无条件导入已从现代 `langchain-community` 删除的 VertexAI 路径，即使项目只使用 OpenAI 也可能导入失败。脚本在导入 Ragas 前安装了一个仅供类型检查的临时兼容模块；不会启用或连接 Google VertexAI，也不会修改 Python 环境中的第三方源码。上游修复发布后可以删除该兼容函数。

# FAQ 离线评测

## 测试集类型

- `exact`：标准问题原句，检查基础 Top-1 与阈值命中。
- `paraphrase`：人工同义改写，检查对真实用户表达的鲁棒性。
- `hard_negative`：包含相似技术词但不属于标准 FAQ 的问题，应该转交 RAG。

## 运行

在 `integrated_qa_system` 目录执行：

```powershell
python evaluation\faq\evaluate_faq.py
```

同时比较多个阈值：

```powershell
python evaluation\faq\evaluate_faq.py --thresholds 0.80 0.85 0.90 0.95
```

只检查测试集引用的标准问题是否存在：

```powershell
python evaluation\faq\evaluate_faq.py --validate-only
```

结果会写入 `evaluation/faq/results/`：

- `faq_details_*.csv`：逐条预测、分数和错误类型。
- `faq_summary_*.json`：总体及分类型指标。
- `faq_report_*.md`：便于阅读和放入项目材料的报告。

脚本还会自动遍历生产 CSV 中全部 467 条标准问题原句，执行全量自检；该结果
单独统计，不会混入 45 条人工黄金集的总体正确率。

## 核心指标

- `top1_accuracy`：预期进入 FAQ 的问题，Top-1 标准问题是否正确。
- `accepted_answer_accuracy`：考虑阈值后，FAQ 是否真正返回了正确答案。
- `false_rejection_rate`：本应由 FAQ 回答，却被转交 RAG 的比例。
- `hard_negative_rejection_accuracy`：本应转交 RAG 的问题是否被正确拒绝。
- `false_acceptance_rate`：硬负例被 FAQ 错误接收的比例。
- `full_corpus_self_check`：全部标准问题原句能否返回自己的答案。

评测前不需要清空 Redis，因为脚本不会读取正式答案缓存。

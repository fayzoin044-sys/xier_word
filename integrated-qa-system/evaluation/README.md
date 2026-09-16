# FAQ 与 RAG 指标评测

该目录只保存离线评测代码、黄金测试集和评测结果，不参与正常问答业务流程。

## 目录

```text
evaluation/
├── faq/                  # FAQ/BM25 路由与匹配评测
│   ├── data/             # 人工标注 FAQ 黄金测试集
│   ├── results/          # 每次运行生成的报告
│   └── evaluate_faq.py   # FAQ 离线评测入口
└── rag/                  # 无虚拟机的 RAG 检索与重排评测
    ├── data/             # 人工标注 RAG 黄金测试集
    ├── results/          # 运行后生成报告与父块清单
    └── evaluate_rag_offline.py
```

## 评测边界

- FAQ：使用确定性指标评测 BM25 Top-1、阈值路由、误接收和误拒绝。
- RAG：当前使用真实本地模型和进程内精确检索评测 ID 指标、消融结果与延迟；后续再按需开启 Ragas 生成质量指标。
- 端到端：FAQ 与 RAG 分别稳定后，再使用混合测试集验证统一入口。

FAQ 评测使用内存数据源和空缓存，复用生产环境的 Jieba 分词、`BM25Okapi`
和 Softmax 计算，但不会连接或修改正式 MySQL、Redis。

RAG 评测复用真实 PDF、父子分块、BGE-M3 和 BGE Reranker，用进程内精确内积
替代 Milvus ANN；不会连接 MySQL、Redis、Milvus，也不会默认调用 DeepSeek。

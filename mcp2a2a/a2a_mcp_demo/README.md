# A2A + stdio MCP 基础示例

天气 Agent 与 RAG Agent 分别提供 A2A 服务，总控根据请求调用它们。每个专业 Agent 启动自己的 stdio MCP 子进程。天气为固定数据，RAG 当前返回“查不到”，用于协议联调。

在 **mcp2a2a 目录**创建独立 Python 3.11 环境后执行：

```powershell
python -m pip install -r a2a_mcp_demo/requirements.txt
$env:DEEPSEEK_API_KEY = "your-api-key"
```

在三个终端中分别设置环境变量并运行：

```powershell
python -m a2a_mcp_demo.weather_server
python -m a2a_mcp_demo.rag_server
python -m a2a_mcp_demo.orchestrator
```

两个 A2A 服务默认使用 8001、8002 端口。`config.py` 中的模型和地址沿用原项目；API_KEY 已改为读取环境变量。本次上传仅完成语法和凭据配置检查，未重跑在线模型调用。

# SmartVoyage v2：A2A + MCP 旅行助手

总控通过 A2A 调用天气、车票、景点、订单四个 Agent；专业 Agent 使用 Streamable HTTP MCP 查询固定模拟数据。支持命令行多轮对话，订单工具不会真实出票、扣票或支付。

在 **mcp2a2a 目录**执行以下命令。先建立独立 Python 3.11 环境，再安装：

```powershell
python -m pip install -r smart_voyage_v2/requirements.txt
$env:DEEPSEEK_API_KEY = "your-api-key"
```

每条服务命令在一个独立终端执行，涉及模型调用的终端均须设置密钥：

```powershell
python -m smart_voyage_v2.mcp_servers.weather_mcp
python -m smart_voyage_v2.mcp_servers.ticket_mcp
python -m smart_voyage_v2.mcp_servers.attraction_mcp
python -m smart_voyage_v2.mcp_servers.order_mcp
python -m smart_voyage_v2.weather_server
python -m smart_voyage_v2.ticket_server
python -m smart_voyage_v2.attraction_server
python -m smart_voyage_v2.order_server
```

最后在另一个终端启动总控：

```powershell
python -m smart_voyage_v2.orchestrator
```

示例问题：`北京天气怎么样？`、`查询北京到上海的车票`。输入 `reset` 清空对话，`exit` 退出。

A2A 服务默认端口为 8101–8104，MCP 为 8201–8204；配置见 `config.py`。模型默认值为原项目配置，可通过 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` 覆盖。本次上传仅完成语法检查，未重跑服务联调。

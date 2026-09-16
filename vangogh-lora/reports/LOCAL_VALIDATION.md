# 初版历史 CPU 验证记录

此文件描述训练前的本地验证状态；后续 GPU 训练与推理产物见项目 README 和 outputs，以下未验证项仅代表当时状态。

验证环境：Windows，Python 3.10.10；本地隔离环境安装 Pillow 11.3.0、datasets 3.6.0、numpy 1.26.4、huggingface-hub 0.34.4。不是服务器 GPU 环境验证。

已通过：

- 原始 zip 全部 7838 张图片完整解码、尺寸统计与 RGB 像素哈希去重。
- 3 项 unittest：重复图片不会跨训练/验证划分；配置参数匹配实际 vendored 官方训练器；推理缩放按 8 对齐并保留宽高比。
- scripts、tests、vendor 的 Python 编译检查。
- smoke 训练命令 dry-run：配置正常解析，所有训练参数均由官方脚本支持。
- infer CLI 帮助入口检查。
- 用官方训练器同样的 datasets ImageFolder 加载真实 358 张训练图，全部可解码、caption 非空、字段为 image/text。
- 真实训练/风格验证集与开发/测试照片集精确像素哈希不相交。

未验证：完整 GPU 依赖共同导入、5090 BF16 实际训练、峰值训练显存、模型保存/恢复的 GPU 执行、LoRA 推理质量、耗时和费用。没有下载 SDXL 或执行训练；没有生成模型权重。

服务器必须按 README 依次运行 preflight、20 步 smoke、权重重载推理、续训检查，再开始正式训练。

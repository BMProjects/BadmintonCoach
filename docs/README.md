# BadmintonCoach 文档索引

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 五层架构、可插拔三支柱、数据契约、两段式标定设计 |
| [MODULE_STATUS.md](MODULE_STATUS.md) | 各模块/后端的实现状态矩阵（就绪 / 已接线 / 待填充）与所需资源 |
| [DECISIONS.md](DECISIONS.md) | 关键设计决策记录（Python 版本、submodule+适配器、标定策略等）|
| [DEVELOPMENT.md](DEVELOPMENT.md) | 环境搭建（uv + Py3.11）、submodule、运行 CLI 与 Gradio 控制台、测试 |
| [TRAINING.md](TRAINING.md) | 球场关键点模型训练（数据集、热图模型、训练结果、世界坐标映射）|
| [ANDROID.md](ANDROID.md) | Android APK 部署规划（端侧/服务器选型、模型导出、许可、是否换 MediaPipe）|
| [EXPERIMENT_LINEHEATMAP.md](EXPERIMENT_LINEHEATMAP.md) | 具名线热图+求交 的多骨干对比实验框架设计（球场标定下一代方案）|

代码内文档：每个 `core/interfaces/*.py` 是该类模块的契约说明；每个 `perception/*/` 适配器文件头部注明上游来源、所需权重与接线状态。

# 设计决策记录 (ADR)

简明记录关键决策与理由，便于后续回溯。

## D1 — 接口化 + 注册表 + 配置驱动的可插拔架构
每个职责定义 `Protocol/ABC`，后端 `@register` 自注册，`build()` 按配置实例化。
**理由**：报告显示每个职责都存在"原型 vs SOTA"多实现；需按平台/性能自由切换而不动上下游。

## D2 — 数据契约用 frozen dataclass
层间只传不可变 dataclass（`core/schemas`）。
**理由**：避免跨层副作用，符合团队不可变编码规范；任一层换实现，契约不变。

## D3 — 上游用 git submodule + 适配器，不改上游源码
TrackNetV3/MonoTrack 等以 submodule 引入 `third_party/`，外层只写适配器翻译 I/O。
**理由**：升级上游零成本；隔离上游的重依赖（如 TrackNetV3 的 pycocotools 仅评测用，适配器内联了 `predict_location` 以规避）。

## D4 — 专用 Python 3.11 环境（uv 管理）
`.python-version=3.11` + `uv.lock`。
**理由**：Py3.13 上 mediapipe/部分生态缺轮子；3.11 与上游兼容性最好。开发用 CPU torch，部署换 CUDA 轮子即可。

## D5 — 姿态统一为 COCO-17
所有姿态后端输出 COCO-17；MediaPipe 33 点在适配器内重映射。
**理由**：BST/RTMPose/报告均以 COCO-17 为准；下游（击球分类）只需面对一种骨架。

## D6 — 羽毛球不交给目标检测器
羽毛球由热图法 `ShuttleTracker`(TrackNetV3) 专责，不进 `Detector` 类别。
**理由**：1–2 像素高速小目标，TrackNetV3 97.5% vs YOLO 53%（报告实测）。

## D7 — 两段式球场标定（固定视角优化）
一次性 bootstrap（解 H + 相机PnP）→ 持久化 `CalibrationProfile` → 后期轻量标志点光流追踪 + 场景切换检测。
**理由**：广播多为固定机位，标定可复用；相机内外参一并求出，为空中羽毛球 3D 重建打底（单应性只覆盖地面平面）。单视角焦距为假设值，标注为可后续精化的近似。

## D8 — 3D 重建诚实报告误差
`ShuttleTrajectory3D` 强制带 `reprojection_error_px`；`null` 后端返回 inf 而非编造点。
**理由**：单目 3D 病态，端到端误差 ~28–37px（MonoTrack 实测）；不得呈现为精确测量。

## D9 — 前端作为可选隔离层（Gradio）
`viz/` + `apps/` 仅依赖数据契约，置于 `[ui]` extra；core 不感知前端。
**理由**：满足"按模块/阶段可视化跟进测试"，又不让前端污染核心、不强加 web 依赖。

# BadmintonCoach 系统架构

羽毛球比赛视频分析工具。核心设计目标：**接口化 + 可插拔**——每个职责（检测、追踪、姿态、标定、3D 重建等）定义统一接口，存在多个可替换后端，按运行平台与性能预算用配置选择，互不影响。

## 分层架构

```
L5 认知层    ASR弱监督 · 时序对齐 · RAG · MADR多智能体 · TTS解说      (Phase 3/5)
L4 长期分析  选手数据库 · 技术指纹 · 雷达图 · 跨场对比                (Phase 4)
L3 指标层    运动学/空间/技术 KPI · rally结构 · 落点热图              (Phase 3)
L2 事件层    击球帧检测 · 击球分类(BST) · rally状态机                 (Phase 2)
L1 感知层    检测 · 球追踪 · 姿态 · 跟踪 · 球场标定 · 3D重建           (Phase 1)  ← 本次已实现
L0 基础设施  视频I/O · 数据契约 · 配置 · 注册表 · 几何                 (贯穿)     ← 本次已实现
```

数据沿层向上流动，每层只依赖下层产出的**数据契约**（`core/schemas`），不依赖具体实现。

## 可插拔机制（三支柱）

1. **接口** (`core/interfaces/`) — 每类模块一个 `Protocol`/`ABC`，定义统一签名。
2. **注册表 + 工厂** (`core/registry.py`) — 后端用 `@register(kind, name)` 自注册，`build(kind, cfg)` 按名实例化。
3. **配置驱动** (`core/config/` + `configs/*.yaml`) — 一个平台一个预设，切换后端/设备/精度只改 YAML。

新增一个后端 = 实现接口 + 加 `@register` 装饰器 + 在 YAML 里点名。上下层零改动。

## L1 感知层后端对照

| 职责 | 接口 | 原型后端 | 生产后端 (SOTA) |
|------|------|---------|----------------|
| 大目标检测 | `Detector` | `yolov8` | `rfdetr` (DINOv2, TensorRT INT8) |
| 羽毛球追踪 | `ShuttleTracker` | — | `tracknetv3` + InpaintNet |
| 姿态估计 | `PoseEstimator` | `mediapipe` | `rtmpose` |
| 球员跟踪 | `PlayerTracker` | `iou` | `botsort` |
| 球场标定 | `CourtCalibrator` | `manual` | `two_stage` / 学习式 `keypoint` |
| 单目 3D 重建 | `Reconstructor3D` | `synthnet` (低延迟) | `monotrack` (物理优化, 高精度) |

各后端的实现状态见 [MODULE_STATUS.md](MODULE_STATUS.md)。

## 两段式球场标定（固定视角优化）

广播多为固定机位，故标定可"算一次、复用全场"：

- **Stage 1 bootstrap**：由 4 个球场角点解地面单应性 H；可选用 PnP 求相机内外参
  （`core/geometry/estimate_camera`）。结果存为 `CalibrationProfile`（JSON，按机位 key）。
- **Stage 2 轻量追踪**：`core/geometry/marker_tracking`（角点光流）+
  `core/io/SceneCutDetector`（HSV 直方图相关性判镜头切换），仅在偏差超阈值时重标。
- **为何要相机参数**：H 只映射地面平面（球员脚步、落点）；空中羽毛球的 3D 需要完整
  相机投影矩阵喂给物理重建器。单视角焦距为假设值（默认=图像宽），可后续精化。

`CourtCalibration` 携带可选 `camera`（`CameraParameters`）字段供 3D 重建使用。

## 可视化与开发者前端（可选层）

- `viz/` — 纯函数叠加层（检测框 / COCO-17 骨架 / 球场重投影+误差 / 球轨迹）+ 标注视频渲染。
- `apps/dev_console.py` — Gradio 控制台：上传视频→选配置→逐模块开关叠加→看标注视频+统计。
- 仅依赖数据契约，置于 `[ui]` extra；**core 不感知前端**。

## 上游开源项目接入

`MonoTrack` / `TrackNetV3` / `BST` 以 **git submodule** 引入 `third_party/`，外层只写**适配器**（`perception/*/`）把上游 I/O 翻译成本项目数据契约。升级上游零成本。submodule 未拉取时，适配器的 `is_available()` 返回 `False`，管线给出明确提示。

```bash
git submodule add https://github.com/qaz812345/TrackNetV3       third_party/TrackNetV3
git submodule add https://github.com/jhwang7628/monotrack        third_party/monotrack
git submodule add https://github.com/Va6lue/BST-...              third_party/BST   # Phase-2
```

## 关键设计约定

- **数据契约用 frozen dataclass**（不可变），避免跨层副作用。
- **依赖按层分组**（pyproject optional-dependencies），边缘部署可裁剪。
- **每个后端实现 `is_available()` + 小样本 smoke test**，平台切换时即时发现缺依赖/缺权重。
- **设备不硬编码 `cuda`**，由配置 `device` 字段决定。
- 坐标系约定见 `core/geometry/court_model.py`（BWF 真实球场坐标，单位米，原点在球场一角，地面 z=0）。

## 目录

```
badminton_coach/
  core/         interfaces/ schemas/ config/ io/ geometry/ registry.py pipeline.py
  perception/   detection/ shuttle/ pose/ tracking/ court/ reconstruction/
  viz/          overlay.py render.py
  cli/
apps/           dev_console.py   (Gradio, [ui] extra)
configs/        singles.yaml   (真实单打片段全链路配置)
third_party/    上游 submodule (TrackNetV3 已拉)
tests/          镜像源码结构
```

# 性能分析与边端部署（Performance & Edge Deployment）

各功能算法的算力需求、系统瓶颈，及边端部署的功能选择依据。

## 测量基准

- 硬件：RTX 4090D（桌面 GPU），输入 1080p，singles 真实转播片段 175 帧（7s @ 25fps）。
- 数字来自 `scripts/profile_pipeline.py` 风格的逐阶段计时（见 git 历史的剖析记录）。

## 逐阶段算力画像

| 阶段 | 调用频率 | 模型/算法 | 参数 | 权重 | 4090D 耗时 | 单帧 | GPU 依赖 | 边端难度 |
|---|---|---|---|---|---|---|---|---|
| 解码 | 每帧 | ffmpeg/OpenCV | — | — | 0.50s | 2.9ms | 否 | 低 |
| 稳定背景 | 每片一次 | 稀疏光流(CPU) | — | — | 1.23s | (一次) | 否 | 低 |
| 球场标定 | 每片**一次** | line_heatmap (efficientvit_b1) | 5.0M | 20MB | 0.14s | (一次) | 是 | 低(可离线) |
| 球场逐帧存在 | **每帧**(可关) | line_heatmap | 5.0M | 20MB | 1.65s | **9.4ms** | 是 | 中 |
| 检测+姿态(合一) | 每帧 | YOLO26n-pose(一次前向出框+关键点) | ~2.9M | 7.6MB | 0.82s | 4.7ms | 是 | **低** |
| 球员跟踪 | 每帧 | iou(numpy) | — | — | 0.01s | ~0 | 否 | 低 |
| ~~独立检测~~ | — | ~~YOLO26n~~ | — | — | **已省** | — | — | — |
| 羽球追踪 | 每帧(滑窗) | TrackNetV3 (U-Net) | 11.3M | 130MB | 0.97s | **5.5ms** | 是 | **高** |
| 击球帧检测 | 每片 | trajectory(CPU) | — | — | ~0 | — | 否 | 低 |
| 击球分类 | **每击球**(稀疏) | BST(transformer) | 1.8M | 7MB | ~ms/拍 | 稀疏 | 是 | 低 |
| 3D 重建 | **每拍**(稀疏) | MonoTrack(scipy 优化) | — | — | ~ms/拍 | 稀疏 | 否 | 中(CPU) |

**整体**：3D-on 全链路 ≈ **28 fps**；2D-only(关 estimate_3d) **检测+姿态合一后 3.23s→2.08s ≈ 84 fps**
(`unified_perception`，省掉独立检测前向)。

## 系统瓶颈（按每帧 GPU 占用排序）

1. **球场逐帧存在 9.4ms**（line_heatmap 前向）——但**可避免**：固定机位只需标定一次
   (`court_profile_path` 缓存)，关掉 `court_per_frame_presence` 即省去整段。
2. **羽球追踪 TrackNetV3 5.5ms** + 130MB——**真正的硬瓶颈**：U-Net 对 (seq_len×3) 通道高分辨
   堆叠卷积，模型大、算力重，是边端实时的最大障碍。
3. **检测 4.7ms ≈ 姿态 4.7ms**——nano 级，已是最省。

去掉可选的逐帧存在后，**固有每帧成本 = 检测+姿态+羽球 ≈ 15ms（67fps）**；TrackNet 占其中 1/3
且最难移植。其余(BST/MonoTrack/标定)都是稀疏或一次性，几乎不影响吞吐。

## 边端部署：功能分层与选择

按"算力代价 vs 价值"分层，指导边端裁剪：

- **Tier 1 必备且便宜**（球员分析：框/骨架/移动/速度）：YOLO26n 检测+姿态 + iou 跟踪。
  全 nano，可导出 TensorRT/NCNN/CoreML/RKNN + INT8，**Jetson Orin/手机 NPU 可实时**。
- **Tier 2 一次性**（度量标定）：line_heatmap 球场标定**离线跑一次**写 profile，使后续
  2D→3D/速度有米制；efficientvit 本就为边端设计，单次成本可忽略。
- **Tier 3 重负载**（羽球轨迹）：TrackNetV3 是瓶颈。边端选项：①降输入分辨率/seq_len；
  ②TensorRT FP16/INT8；③降帧率跑(羽球可插值)；④弱端直接**离线/云端**算轨迹。
- **Tier 4 稀疏便宜**（事件层）：BST 击球分类(1.8M, 仅击球时跑)、MonoTrack 3D(每拍 scipy)
  ——量小、事件触发，**任何端都不构成负担**。

### 按目标平台

| 平台 | 建议 |
|---|---|
| 桌面/服务器 GPU | 全链路实时；本项目现状 |
| Jetson Orin Nano/NX | YOLO→TensorRT-INT8；TrackNet→FP16+降分辨/降帧；球场离线标定；BST/3D 照常。预期降帧率近实时 |
| 手机(NPU/CoreML/NNAPI) | YOLO 检测+姿态+速度本地实时；TrackNet 改轻量或云端；BST/3D 本地 CPU |
| 纯 CPU | 仅 YOLO-nano 可近实时；TrackNet/3D 转离线批处理 |

## 已内建的性能杠杆

- `perception.estimate_3d`（开关 2D→3D 子系统）：关 → 纯 2D 快链路 ~2.7× 提速。
- `perception.court_profile_path`（固定机位标定缓存）：整场只标一次。
- `CourtCalibrator.present_frames`（批量前向）+ 输出抽帧到 15fps。
- BST/MonoTrack 事件触发（非每帧）。

## 后续优化方向（优先级）

1. **TrackNetV3 边端化**（最大收益）：蒸馏/降分辨/INT8，或评估更轻的 shuttle 追踪器。
2. **去逐帧球场存在**：固定机位用一次标定 + 场景切换检测代替逐帧前向。
3. ✅ **检测+姿态合一**(已实现 `unified_perception`)：YOLO26-pose 一次前向出框+骨架，
   省独立检测前向，2D-only 提速 ~1.55×，且框↔姿态精确配对(免 IoU 匹配)、跟踪更干净。
4. **统一 INT8 导出**：YOLO/efficientvit/TrackNet 走 TensorRT/ONNX-Runtime INT8。

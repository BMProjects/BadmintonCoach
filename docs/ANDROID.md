# Android APK 部署规划（仅清单，未执行）

把现有功能（球员动作识别、羽毛球检测跟踪、球场 2D 定位、球员/羽毛球 3D 坐标）打包到
Android。本文列出所需工作、选型与取舍。**结论先行**：

- **不需要"用 Java 重写"**。模型**导出**到移动推理引擎（不是重写）；**胶水算法**（单应性、
  白线精修、物理 3D、跟踪、状态机）若上端侧则用 **Kotlin + C++/NDK(OpenCV)** 重写，**Kotlin 优先**，不是 Java。
- **是否换 MediaPipe**：**端侧姿态推荐换** MediaPipe Pose / MoveNet(TFLite)（移动端最成熟、有 GPU delegate）；
  其余自训模型（球场 MobileNetV4、TrackNet、BST）导出到 TFLite/NCNN/ONNX-Runtime-Mobile 即可。
- **最省力路径**：先做**客户端-服务器**（手机录制/上传，服务器跑现有 Python 管线），**零重写**；
  再把轻量模块逐步搬到端侧实现实时。

---

## 0. 首要决策：端侧 vs 服务器

| 方案 | 重写量 | 实时性 | 离线 | 适用 |
|------|-------|-------|------|------|
| **服务器推理**（推荐起步）| 几乎为零（现有 Python 直接用）| 受网络/服务器 | 否 | 快速出 demo、重模型(TrackNet/MonoTrack)友好 |
| **端侧推理**（终态目标）| 大（胶水重写 + 模型导出）| 可实时 | 是 | 隐私、无网、产品化 |
| **混合**（推荐演进）| 中 | 轻量模块端侧实时、重模块上云 | 部分 | 平衡 |

> 重模型（TrackNetV3 多帧+中值背景、MonoTrack 的 scipy 非线性优化）端侧实现成本高，
> 建议先留服务器，端侧先上**球场(MobileNetV4)+姿态(MoveNet)+球员检测**。

## 1. 各模块的端侧化工作

| 模块 | 现状 | 端侧方案 | 工作量 |
|------|------|---------|-------|
| 球场 2D 定位 | MobileNetV4 热图(自训, 15MB) | 导出 ONNX→TFLite/NCNN，INT8 量化 | 小（已轻量）|
| 白线精修 | OpenCV ICP(Python) | OpenCV Android(C++/Kotlin) 重写 | 小-中 |
| 球员检测 | Ultralytics YOLOv8 | **换**（AGPL，见许可）：MediaPipe/MoveNet 人体框 或 自训 nano 检测器→TFLite | 中 |
| 球员姿态 | RTMPose(onnxruntime) | **换 MediaPipe Pose / MoveNet**(移动端最优) 或 RTMPose→ncnn | 中 |
| 球员跟踪 | IoU(Python) | Kotlin 重写（IoU/ByteTrack）| 小 |
| 羽毛球追踪 | TrackNetV3(多帧热图) | 导出 ONNX→TFLite；改**流式**(滑窗+在线中值背景) | 大 |
| 击球动作(BST) | 未实现(L2) | 先实现再导出 | 大（含 L2）|
| 3D 重建 | MonoTrack(scipy 优化) | C++ 重写 LM/最小二乘 + 物理 ODE（无 scipy）| 大 |
| 坐标变换/指标 | Python | Kotlin 重写 | 小 |

## 2. 推理引擎选型（端侧）

- **TFLite**：Android 一等公民，NNAPI/GPU delegate，MediaPipe 生态；推荐主力。
- **NCNN / MNN**：纯 C++、移动端极致优化，适合自训 CNN（球场/TrackNet）。
- **ONNX Runtime Mobile**：从现有 ONNX 直接来，统一；delegate 支持。
- **ExecuTorch**（PyTorch 端侧新方案）：若想留 PyTorch 链路。
- 建议：自训模型走 **ONNX→(TFLite 或 NCNN)** + INT8 量化；姿态用 MediaPipe/MoveNet 现成 TFLite。

## 3. 打包前的优化工作（端侧通用）

1. **模型导出+量化**：每个模型 ONNX 导出→INT8/FP16 量化→在目标机测精度/时延。
2. **算子兼容**：检查导出后算子是否被 TFLite/NNAPI 支持（如 argmax 解码可放 CPU 后处理）。
3. **TrackNet 流式化**：去掉"整段 median 背景"，改在线滑窗中值/EMA 背景，定长 seq 输入。
4. **检测器去 NMS 依赖**或用端侧 NMS；输入分辨率下采样到 320/384。
5. **3D 物理优化轻量化**：限制迭代次数 / 用解析初值，C++ 实现，避免 scipy。
6. **多线程/异步**：相机帧→推理→渲染分线程，避免阻塞 UI；GPU delegate。
7. **APK 体积**：模型 INT8、ABI 拆分(arm64-v8a)、按需下载大模型。
8. **内存/热**：定长缓冲、复用张量、控制并发模型数；监测发热降频。

## 4. Android 工程脚手架（端侧）

- Android Studio + **Kotlin**，minSdk 26+，**NDK** 用于 OpenCV/C++ 胶水。
- 依赖：OpenCV Android SDK、TFLite/ONNX-Runtime-Mobile/NCNN AAR、CameraX。
- 模型作为 `assets/`（或首启下载）。
- 结构：`camera → frame → (court/det/pose/shuttle 推理) → 胶水(单应性/3D/跟踪) → Canvas/OpenGL 叠加`。
- 推理在后台线程；结果回主线程渲染。

## 5. 许可合规（重要，打包前必查）

- **Ultralytics YOLOv8/11 = AGPL-3.0** → 闭源 APK 不能直接分发，需商业授权或换模型（MediaPipe/自训）。
- TrackNetV3 / MonoTrack / BST / RTMPose 各自许可需逐一确认可再分发（多为 MIT/Apache，但要核实权重）。
- 数据集（Roboflow CourtDet 等）许可影响训练产物的分发权。

## 6. 是否"用 Java 重写"？是否换 MediaPipe？——直接回答

- **不必用 Java**。端侧用 **Kotlin（UI/逻辑）+ C++/NDK（OpenCV、3D 优化）**；模型是**导出**不是重写。
  若选服务器方案则**完全不用重写**（现有 Python 加个 FastAPI/gRPC 接口即可）。
- **MediaPipe**：**姿态强烈建议在端侧改用 MediaPipe Pose 或 MoveNet**（移动端时延/功耗最优、API 现成）；
  检测可用 MediaPipe 或移动 nano 检测器以**规避 Ultralytics AGPL**；球场/球追踪是自训 CNN，
  导出到 TFLite/NCNN 即可，不必换成 MediaPipe。

## 7. 建议路线

1. **M1 服务器版**：现有 Python 管线 + FastAPI；Android 仅相机+上传+结果叠加（最快可用，零算法重写）。
2. **M2 端侧轻量**：球场(MobileNetV4)+姿态(MoveNet)+球员检测 上端侧 TFLite，实时 2D 分析。
3. **M3 端侧完整**：TrackNet 流式 + 3D(C++) + L2 动作，全端侧。

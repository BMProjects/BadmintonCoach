# 模块实现状态矩阵

状态说明：
- **就绪** — 无外部依赖，开箱即用（基线/功能性实现）。
- **已接线** — 适配器已写好并验证推理链路，需提供权重/依赖即可真实运行。
- **待填充** — 已注册并占位，`is_available()` 守护，逻辑尚未实现。

## L1 感知层

| 职责 | 后端 | 状态 | 依赖 / 所需资源 |
|------|------|------|----------------|
| 检测 | `null` | 就绪 | 无（返回空） |
| 检测 | `yolo` | 已实现 ✅(YOLO26, singles.yaml 默认) | Ultralytics 通用；singles.yaml 用 yolo26n.pt(NMS-free)；需 `class_map` |
| 检测 | `rfdetr` | 已接线 | `rfdetr` 包 + 羽毛球域微调权重；需 `class_map` |
| 球追踪 | `null` | 就绪 | 无 |
| 球追踪 | `tracknetv3` | 已实现 ✅(真权重 GPU 实跑，样例误差 ~1-3px) | submodule(已拉) + `[shuttle]` extra + `weights/ckpts/TrackNet_best.pt`(已下载) |
| 球追踪 | `replay` | 就绪 ✅ | 回放 sidecar JSON 的 2D 轨迹（开发/demo 用，非检测器）|
| 姿态 | `null` | 就绪 | 无 |
| 姿态 | `yolo_pose` | 已实现 ✅(YOLO26-pose, ~206FPS, singles.yaml 默认) | Ultralytics YOLO26-pose 原生 COCO-17，单阶段；按 IoU 匹配到球员框；全程 YOLO/torch，**无 onnxruntime** |
| 球员跟踪 | `iou` | 就绪 ✅(coasting+距离+碎片合并) | numpy；轨迹保活(max_age)跨漏检 + 中心距离关联 + **碎片合并后处理**(链接时序不重叠、空间连续的片段,跨遮挡/回放>max_age);移动统计另设**物理速度上限(9m/s)剔除 id-switch 瞬移**,单打稳定 |
| 球员跟踪 | `botsort` | 已实现 ✅(ReID, 解决交叉) | `boxmot` 19 BoT-SORT + OSNet ReID(权重自动下载)；**接口传 `frames`**做外观重识别→**彻底解决穿插/近网 id-switch**;为羽球调参(track_high 0.35/new 0.45/buffer 75)→样例 2 干净轨;preset `singles_botsort.yaml` / 前端 Tracker 下拉 |
| 球场标定 | `line_heatmap` | 已训练 ✅(efficientvit_b1, 中位 0.57px, singles.yaml 默认) | 12 具名线热图→逐线拟合求交还原 22 交点→DLT 单应 + PnP 相机；亚像素、抗大变形；**误检门控**：min_overlap(白线重合)+min_observed(画面内置信点)+reproj→拒绝无/不完整球场；court_lines_evit_b1.pt(5M/20MB) |
| 球场标定 | `auto` | 已实现 ✅(择优) | 同时跑 line_fit + line_heatmap，取**全模型白线重叠更高者**，互为回退（configs/singles_auto.yaml）|
| 球场标定 | `line_fit` | 已实现 ✅(传统几何法) | 多色面门控→**局部 top-hat 细白线提取(去地面反光/大亮斑)**→Hough→候选角点组合→全 BWF 模型重叠打分→精修；广播重叠 0.97 |

> **标定鲁棒化**：管线先用 `core/io/stable_background`（每200ms抽帧+稀疏光流找稳定窗口→中值背景去运动人员）得到干净球场图，再在其上标定（避开坏首帧/字幕/遮挡）。是 2D→3D 的唯一度量基础。
| 球场标定 | `two_stage` | 就绪 ✅ | 手标 4 角点 + 相机PnP + profile 持久化（固定机位复用）|
| 球场标定(复用) | `perception.court_profile_path` | 已实现 ✅ | 任意后端通用：管线级缓存 CalibrationProfile；命中则跳过 stable_background+标定，**整场只标一次**跨多片复用 |
| 球场标定 | `manual` | 就绪 | 4 角点 config |
| 3D 重建 | `null` | 就绪 | 无（诚实返回空）|
| 3D 重建 | `monotrack` | 已实现 ✅(已验证恢复已知阻力轨迹) | `scipy`；MonoTrack 物理优化法移植；需 `court.camera`。焦距由球场两正交灭点估计(`estimate_focal_from_court`，近正交视角退回画面宽先验) |
| 3D 重建 | `synthnet` | 待填充 | 合成数据训练的 FNN/LSTM 权重 |

## L2 事件层

| 职责 | 后端 | 状态 | 说明 |
|------|------|------|------|
| 击球帧检测 | `trajectory` | 已实现 ✅ | 2D 轨迹方向反转检测；真实片段 9 hits/7s；无权重 |
| 击球帧检测 | (HitNet) | 待接 | 学习式，可换 |
| 击球分类 | `heuristic` | 已实现 ✅(粗分类, 备选) | 由分段 3D 弹道形状判 smash/clear/drop/drive/lift；无权重、轻量备选(BST 不可用时) |
| 击球分类 | `bst` | 已实现 ✅(ShuttleSet 真值 yolo_pose 粗分 ~60% / 细类型 ~47%) | 输入管线对上游源码逐函数验证全等(test_bst_port)。**关键修复：每拍取以击球帧为中心的窗口**(上游 between_2_hits_with_max_limits)——原错用击球后飞行段漏掉挥拍。修后 match1 200 拍：粗分(7类)**59.6%**、细类型(17类)**47.4%**(31.9%→见下)；远高于随机(粗 14%/细 6%)。剩余差距=姿态质量/分段精度/检测覆盖，非时窗 bug。det 阈值 0.3→0.15 使 unknown 27%→22% |
| 事件编排 | `events.analyze_events` | 已实现 ✅ | 检测击球→**按击球分段做 3D**→分类→**rally 分段 + 战术统计** |
| 战术统计(L3) | `events.rally`/`events.stats` | 已实现 ✅ | rally 状态机(按击球间隔分段)；MatchStats：回合数/每回合拍数、stroke 分布、球员移动距离+速度(单应性, 取最长2轨+平滑+P95 max)、击球接触点、**真实落点(分段 3D 弹道 z=0 降交/抛物外推)** |
| 生物力学(L3) | `biomechanics` / `pose2d` | 已实现 ✅(MVP, 可插拔) | 2D pose + `PlayerProfile`(身高/体重/利手/性别)→ 每拍racket侧关节角(肩/肘/髋/膝)、角速度、**I·α 关节力矩代理**(de Leva/Winter 人体测量按身高体重缩放)、**动力链时序**(髋→躯干→上臂→前臂)。平面/相对近似 |
| 生物力学(L3) | `biomechanics` / `lift3d` | 已实现 ✅(P2, 学习式 3D) | **单目 2D→3D 抬升**后算 **3D 关节角 + 3D I·α + 3D 动力链**。默认 `lifter=motionbert`(**MotionBERT 3D ONNX**, COCO→H36M→crop_scale→DSTformer):`seq`=27/81/**243**,默认 **81**,**以击球帧为中心取 seq 帧上下文窗口**跑模型(更长上下文→更稳的 3D),仅取击球区间帧算指标;权重 HF 自动下载;缺权重/ort 回退 `lifter=analytic`(人体测量骨长,无依赖) |

## 可视化增强
- **速度标注就近显示** `viz/metrics.py`：**球员速度(m/s)标在球员框上方**、**羽毛球速度(km/h)标在球附近**（描边可读）。球员脚点经单应性→米→m/s；羽毛球速度由分段 3D 求。
- **球员检测框默认关闭**（仅骨架）。
- **输出抽帧到 15fps**：`render_video(output_fps=15)` 默认，按时间戳抽帧、正确播放时序。
- **3D 估计可开关** `perception.estimate_3d`（默认 true）：把 stable_background + 球场标定 +
  逐帧 presence + 3D 重建归并为一个子系统，`pipeline.run(estimate_3d=...)` 整体开关；前端
  dev_console 有 “3D estimation” 复选框 + L1/L2/render 计时显示，关掉即纯 2D 流水线
  (检测/姿态/跟踪/球2D)，样例上 ~2.7× 快，便于对比不同流水线耗时。
- **逐帧 presence 批量化** `CourtCalibrator.present_frames`（line_heatmap 一次 GPU 批前向）；
  注：小骨干前向为算力瓶颈(~9ms/帧)，批量本身收益有限，主要提速来自移除逐帧白线 overlap。
- **时域事件标注** `viz/overlay.draw_event_hud` + `render._event_lookups`：视频上叠加
  「Rally r · Shot k/N · STROKE」横幅 + 击球帧黄环 HIT 闪烁（按时长保活以抗抽帧）；前端
  summary 输出**带时间戳的击球时间线**（#i  t0-t1s  stroke  conf）。`render_video(analysis=ma)`。
- **落点**：真实落点(z=0)在 summary/报告中给出坐标(视频上不再常驻红 X,避免遮挡);`draw_landings` 保留备用。
- **前端布局**：input + settings 移入**左侧 `gr.Sidebar`**;主区为 Studio(播放器+时间线)/ Player Report 标签。
- **渲染细化**：羽球轨迹改 **~1s 连线**(`draw_shuttle` polyline);击球 **HIT 仅短闪**(0.15s,不常驻);
  stroke 类型**标在击球球员旁**(`draw_stroke_label`,按类型配色);生物力学**力量用躯干线颜色表示**
  (`draw_trunk_force`,绿→红 = 低→高,不再标关节角文字);时间线新增 **Effort 行**(每拍负荷条)。
- **球员档案输入**：dev_console 增 身高/体重/利手 输入 → 喂 biomechanics 缩放与归一化。
- **球员报告页** `viz/report.build_player_report` + dev_console "Player Report" 标签:Markdown 报告
  (档案/回合/stroke 分布/移动/**生物力学:逐关节负荷、动力链合格率、峰值发力拍、逐拍表、教练建议**)。
- **前端 stroke 分类器可选** dev_console 下拉 bst/heuristic（bst 为默认，ShuttleSet 验证）。
- **剪辑器式前端布局** `apps/dev_console.py` + `viz/timeline.render_timeline`：处理后**全宽播放器
  自动播放标注视频**(自带 scrubber 拖动)，下方**全宽多轨事件时间线**(Rally/Hit/Stroke 三行沿
  时间轴，按 stroke 类型配色 + 图例 + 秒标尺);**点击时间线跳转视频**(像素→时间→JS 设 currentTime)；
  设置/统计折叠到 accordion，不再预览原始视频。
- **检测+姿态合一** `perception.unified_perception`(默认开)：YOLO26-pose 一次前向出框+骨架，
  省独立检测前向(2D-only ~1.55× 提速)，框↔姿态精确配对、跟踪更干净；非 YOLO-pose 后端自动回退。
- **球场识别用红色线**、**动作(姿态)识别用黄色** `viz/overlay.py`。
- **球场诊断**：`python -m scripts.diagnose_court assets` → 在图上画检测白线(青)+匹配球场
  (绿=采纳/红=拒绝)+关键点(黄)，输出 `assets/court_diag/`，用于排查标定失败原因。

**关键修复**：按击球分段重建 3D 把整段单抛物线拟合的 **238px → 4.2px**（每拍一条弹道）。

## L0 基础设施（全部就绪）

数据契约、接口、注册表/工厂、配置(YAML+pydantic)、视频I/O、滑窗、几何（单应性/相机PnP/羽毛球阻力ODE/球场模型）、场景切换检测、标志点光流追踪、Phase-1 管线编排。

## 可视化 / 应用（就绪）

`viz/`（检测框/COCO-17骨架/球场重投影/球轨迹叠加 + 标注视频渲染）、`apps/dev_console.py`（Gradio 开发者控制台，`[ui]` extra）。

## 真实单打片段（assets/sample_singles.mp4，configs/singles.yaml）

L1+L2 全链路在真实转播上已跑通：**line_heatmap 球场标定**(reproj 1.1px + 相机, 68 帧有球场)
→ YOLO26 球员检测(court 过滤后恰 2 人) → YOLO26-pose 姿态(GPU) → TrackNetV3 球追踪
(166/168 帧可见) → 按击球分段 3D → heuristic 击球分类(drop/smash 多样)。

## 已完成（本轮系统搭建）

- ✅ **球员跟踪稳定化**：iou coasting + 距离关联（14→2 主轨）。
- ✅ **焦距标定提精**：球场两正交灭点估焦距，近正交退回先验。
- ✅ **固定机位复用**：`court_profile_path` 管线级缓存，整场只标一次。
- ✅ **rally 状态机 + 战术统计**：rally 分段 + MatchStats（拍数/stroke 分布/移动/落点）。
- ✅ **真实落点(z=0)**：分段 3D 弹道 z=0 降交/抛物外推 → 球场 xy + 红 X 可视化。
- ✅ **时域事件标注 + 前端展示**：视频 HUD + 击球时间线 + 分类器下拉。
- ✅ **BST 校准 + ShuttleSet 真值验证**：下载 2 场视频(`scripts/download_shuttleset.py`，
  破 YouTube cookies+PO token+nsig 三层反爬)→ `scripts/validate_bst_shuttleset.py` 验证。
  **定位主因并修复：每拍时窗应以击球帧为中心(±1.5s)而非取击球后飞行段**——修后 yolo_pose
  粗分 **31.9%→59.6%**、细类型 47.4%；**两场交叉验证一致**(match2 粗 67.2%/细 50.0%)。
  证明 yolo_pose≈RTMPose，主差距是时窗 bug；剩余差距=姿态质量/分段/检测覆盖(unknown ~23%)。
- ✅ **性能/边端分析** `docs/PERFORMANCE.md`：逐阶段算力画像、瓶颈(TrackNetV3 + 逐帧球场存在)、
  边端功能分层(Tier1 检测+姿态可实时 / Tier3 羽球追踪需优化)。
- ✅ **生物力学 MVP(L3)** 可插拔 `biomechanics/pose2d`：2D pose + 身高/体重 → 关节角/角速度/
  I·α 力矩代理/动力链时序；前端力量用**躯干线颜色**(绿→红)、时间线 Effort 行。
- ✅ **生物力学 P2: `lift3d` 学习式 3D**：**MotionBERT 3D ONNX**(权重 HF 自动下载,onnxruntime 跑)
  → 真 3D 关节角 + 3D I·α + 3D 动力链;缺权重回退 analytic。前端可选 lift3d/pose2d。WHAM/OpenSim=P3。
- ✅ **球员报告页**：dev_console "Player Report" 标签 + `viz/report`,Markdown 教练报告。
- ✅ **MotionBERT 81/243 + 上下文窗口**：以击球帧为中心取 seq 帧上下文(默认 81)跑 MotionBERT,
  更稳的 3D;前端可选 seq。**球员跟踪碎片修复**:碎片合并后处理 + 移动统计物理速度上限剔瞬移。
- ✅ **botsort+ReID 解决交叉 id-switch**：boxmot19 BoT-SORT,接口传 frames 做 OSNet 外观重识别,
  调参后样例 **2 干净轨**;`PlayerTracker.track(dets, frames)` 接口扩展;preset + 前端 Tracker 下拉。

## 下一步建议优先级（剩余工作）

1. **业余/低角度球场泛化**（最大缺口）：现有学习型标定只在广播分布(court_kp_official)训练，
   对业余红/蓝/绿胶地、大斜角、出框球场不泛化（diagnose_court 实测全失败，误检门控已让它
   "拒绝"而非"画错"）。需在代表性业余数据上微调 efficientvit_b1（数据收集/自标注待定）。
2. **BST 提残余**：yolo_pose 已达 73%(≈论文)；剩 ~27% 段因 detection 漏检判 unknown，
   修检测覆盖(逐帧稳 2 人)可进一步提可判率；细分 35 类评测可选。
3. **多帧焦距 / 内参精修**：单帧灭点法对近正交视角退化，多帧聚合或加畸变模型可再提精。

# 球场关键点模型训练

学习式球场标定（`heatmap` 后端）的训练流程。数据集软链接在 `data/`，原始数据在
`/home/bm/Data/CourtDet/`。

## 数据集

`/home/bm/Data/CourtDet/` 下收集了若干 COCO 关键点数据集（Roboflow 导出）+ KSeq：

| 数据集 | 关键点 | 图片 | 说明 |
|--------|-------|------|------|
| BadmintonCourtDetectionOffical.v5 | **22**（球场线交点全集） | 918/202/74 | 单一 schema、点位分离、数据最多 → **训练所用** |
| Badminton Court Keypoint Dataset v3/v4/v5 | 30（含角点簇 tl-1..tl-4） | 合并 195 | 同 schema 可合并，但密集簇点局部难分辨 → 实测 PCK 仅 0.26 |
| KSeq_train/test | — | 仅 JSON，无图片 | 不可用于检测器训练 |

软链接：`data/courtdet`→原始目录；`data/court_kp`→合并后的 30-kp 集；
`data/court_kp_official`→22-kp 官方集。

合并 30-kp 家族：`python -m scripts.merge_court_datasets`（去重、重切分 80/10/10）。

## 模型与方法

主流热图回归（SimpleBaselines / TennisCourtDetector 同类）：主干 + 3 层转置卷积
上采样到 1/4 分辨率 + 1×1 头输出每关键点一张热图；Gaussian 目标 + 掩膜 MSE；
argmax 解码。主干可换（`--backbone`）：默认 **MobileNetV4-conv-small**（2024 最新
轻量骨干，~1.3M 参，面向边缘实时）；另有 `resnet34`（更重更准变体）。代码见
`training/court_kp/`。

### 无球场负样本（解决"幻影球场"）

纯关键点模型对任意画面都会输出峰值，导致无球场/不完整画面生成"幻影球场"。
负样本（无球场图，schema 无关）以**全零目标热图 + 全通道监督**加入训练，教会模型
无球场时低响应；推理端按置信度阈值（≥4 个高置信关键点）拒绝。负样本由
`python -m scripts.collect_negatives` 从各数据集的未标注图收集（99 张）。

## 训练（硬件优化）

针对 R5 7500F(12线程) + 64GB + 4090D：数据 186MB 全部**预读入 RAM**（worker 不碰磁盘）、
热图用**预算高斯核粘贴**（免逐点 meshgrid）、`DataLoader` **10 workers + pin + persistent +
prefetch**、**AMP bf16 autocast**、`cudnn.benchmark`。GPU 利用率从 11% 升到 46–99%
（小模型小数据下本就 burst 式，详见下"利用率说明"）。

```bash
python -m training.court_kp.train --data data/court_kp_official --val-split valid \
    --negatives /home/bm/Data/CourtDet/negatives.json \
    --backbone mobilenetv4_conv_small --input 384 --epochs 80 --batch 48 \
    --workers 10 --sigma 3 --out weights/court_kp_mbv4.pt
```

**结果（官方 22-kp 测试集，MobileNetV4 + 负样本）**：
median px 11.2 / PCK@0.02 0.98 / **PCK@0.05 1.00** / court_recall **1.00** /
**neg_reject 1.00**（无球场帧 100% 拒绝）。模型 **3.76M 参 / 15MB**。真实片段多帧自举
标定 reproj ~10px，全球场线叠加对齐。

> **超参注意**：`--sigma` 要与热图分辨率匹配。heatmap=input/4；sigma=2 在 96×96 上
> 目标过小，MSE 会塌缩成全零（实测训崩）；用 **sigma≥3**。input 512 时可 sigma 3。

> **GPU 利用率说明**：小模型(3.76M)+小数据(~1k)下，单 batch 在 4090 上几毫秒算完，GPU
> 频繁等数据 → 占用 burst 式波动（0%↔99%）正常，非配置问题；此任务 GPU 非瓶颈。想拉满
> 需更大模型/分辨率（与边缘轻量目标冲突），故不强求。

## 幻影球场修复（逐帧门控）

纯关键点模型对任意画面都输出峰值，且管线"标定一次复用"会在**每帧画同一球场框**——
无球场/不完整帧上即"幻影球场"。修复：① 负样本训练让无球场时低响应；② `CourtCalibrator.is_present`
逐帧判断（≥4 高置信关键点），`PerceptionResult.court_frames` 记录有球场的帧，渲染只在这些帧画
球场。配置 `perception.court_per_frame_presence: true`（singles.yaml 已开）。实测无球场/转场/
不完整特写帧均不再叠加球场框。

## 世界坐标映射（供单应性使用）

模型输出 22 个图像关键点；其世界坐标（BWF 米）由 line_fit 在 178 张官方图上自动反推
中位值得到，存于 `weights/court_kp_official_world.json`（与 BWF 几何吻合到厘米级）。
推理后端 `heatmap` 用置信关键点 + 世界坐标 RANSAC 求单应性（+相机 PnP）。

## 让业余/手机视角也能标定（数据收集 + 增广）

广播模型在业余手机视频（低斜角、线淡、多场地）上检测不到球场。两条路：

1. **固定机位手机视频：4 角点即可**（无需训练）。用标注工具点 4 个球场角点 →
   `manual`/`two_stage` 后端立即标定整段（固定机位复用）。
2. **让模型自动泛化：收集业余样本重训**。
   - **4 角点 → 22 关键点自动标注**（球场几何固定，22 点世界坐标已知）：
     `python -m scripts.label_from_corners --video clip.mp4 --corners "nLx,nLy nRx,nRy fRx,fRy fLx,fLy"`
     固定机位下每帧标注一致，几次点击得到整段 22-kp 训练数据。
   - **可视化标注工具**：`python -m apps.annotate_court`（0.0.0.0:7861）——上传视频、
     点 4 角、预览投影全场线核对、一键 `Label video` 导出 COCO 到 `data/amateur_court/`，
     或 `Copy manual corners` 直接拿去配置里即时标定。
   - **透视增广（opt-in）**：训练加 `--perspective` 启用随机单应性 warp，模拟不同机位/
     斜角。⚠️注意：**仅对纯广播数据单用透视增广会略降广播精度**（实测中位 11→15px、
     PCK@0.02 0.98→0.95）且对业余的增益**未经验证**——所以默认**关闭**，权重 court_kp_mbv4.pt
     仍是更准的非增广模型；透视增广**应在并入真实业余标注后**再启用。
   - 合并业余集与官方集后重训（`--data` 指向合并集，加 `--perspective`）。多段不同场馆/
     角度的业余视频越多，泛化越好。变体 court_kp_mbv4_persp.pt 为透视增广参考权重。

## 推理（已接入 court_calibrator）

`configs/singles.yaml` 默认 `court_calibrator.backend: heatmap`。真实片段实测
reproj ~11px、相机解出、全球场线叠加对齐。无权重时回退 `line_fit`。

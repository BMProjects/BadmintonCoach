# 开发指南

## 环境搭建（uv + Python 3.11）

```bash
cd BadmintonCoach
uv venv --python 3.11            # .python-version 已固定为 3.11
source .venv/bin/activate

# 核心依赖
uv pip install "numpy>=1.24" "opencv-python>=4.8" "pydantic>=2.6" pyyaml

# 按需安装后端依赖组（见 pyproject [project.optional-dependencies]）
uv pip install -e ".[pose,ui,dev,reconstruction]"   # RTMPose + Gradio + 测试 + scipy
# torch CUDA 轮子（RTX 4090：driver 支持 cu12x；CPU 机器改 whl/cpu）
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[shuttle]"          # TrackNetV3 上游运行依赖(parse/pandas)
# GPU 姿态：用 onnxruntime-gpu（CPU 版没有 CUDAExecutionProvider）
uv pip uninstall onnxruntime && uv pip install onnxruntime-gpu
# 验证 GPU：python -c "import torch; print(torch.cuda.is_available())"
# RTMPose 走 CUDA 由 core.cuda_bootstrap 自动加载 torch 自带的 CUDA 库，无需手动设 LD_LIBRARY_PATH

uv lock                                  # 生成/更新 uv.lock
```

## 上游 submodule

```bash
git submodule update --init --recursive
# 已加入：third_party/TrackNetV3
# 待加入：third_party/monotrack（3D 重建时）
```
## 模型权重

TrackNetV3 权重（Google Drive，132MB，含 TrackNet_best.pt + InpaintNet_best.pt）：
```bash
uv pip install gdown
gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O weights/TrackNetV3_ckpts.zip
cd weights && unzip TrackNetV3_ckpts.zip && cd ..    # -> weights/ckpts/TrackNet_best.pt
```
路径已写入 `configs/singles.yaml` 的 `shuttle_tracker.weights`。RTMPose / YOLOv8
权重首用自动下载，无需手动准备。

## 运行

```bash
badminton backends                                                   # 列出已注册后端
badminton analyze --video assets/sample_singles.mp4 --config configs/singles.yaml
```

## 启动前端（Gradio 开发者控制台）

两种方式（任选其一）：
```bash
# 方式 A：模块运行，无需安装本项目
python -m apps.dev_console

# 方式 B：先把项目装为可编辑包注册命令，再用短命令
uv pip install -e .
badminton-console
```
默认监听 `http://0.0.0.0:7860`（改端口：`GRADIO_SERVER_PORT=7870 python -m apps.dev_console`）。

**正确退出服务（释放端口）**：
- 前台运行：在该终端按 **Ctrl+C** —— 已做干净关闭（`demo.close()` 释放端口）。
- 后台/nohup 运行：`pkill -f apps.dev_console`（或 `kill <PID>`）。
- 若启动报 `Cannot find empty port ... 7860`：说明上一个实例还在跑，先用上面命令停掉再启动
  （程序会给出同样的提示，而非崩溃）。

打开后操作：上传 `assets/sample_singles.mp4` → 选 `configs/singles.yaml` → 勾选叠加模块 → Run。

## 配置预设

- `configs/singles.yaml` — 真实单打片段全链路：line_fit 自动球场标定(+相机) + YOLOv8
  球员检测(court 过滤) + RTMPose(GPU) + TrackNetV3 + MonoTrack 3D。当前唯一维护的配置。
  其它平台/后端组合按需复制此文件修改对应 `backend` 字段即可。

切换某模块后端 = 改对应 `backend` 字段一行；自定义参数直接写在该块下（pydantic `extra=allow`）。

## 测试

```bash
pytest -q                      # 全部
pytest --cov=badminton_coach   # 覆盖率
ruff check badminton_coach apps tests
ruff format badminton_coach apps tests
```

## 新增一个后端

1. 在 `badminton_coach/perception/<kind>/` 新建文件，实现对应 `core/interfaces` 接口；
2. 加 `@register("<kind>", "<name>")`，并在该子包 `__init__.py` 导入以触发注册；
3. 实现 `is_available()`（检查依赖/权重/submodule）；
4. 在 YAML 配置把 `backend` 指向 `<name>`。无需改动 pipeline 或其它后端。

# YOLO11n 单类人脸检测训练复现指南

本文记录 Windows + RTX 5060 环境下，使用 WIDERFace 与 MAFA 难例训练 YOLO11n 单类人脸检测器，并为后续 ONNX/RKNN 部署准备模型的流程。所有数据路径、参数、依赖版本应以实际机器检查结果为准；不要把本次实验结果直接视为最终部署指标。

## 1. 实验目标与数据划分

- 任务：单类人脸检测，类别名 `face`，类别 ID 为 `0`。
- 模型：Ultralytics YOLO11n，从 PyTorch `.pt` 权重训练，训练完成后再导出 ONNX。
- 输入尺寸：`640×640`。
- 目标数据集：`D:\face_train\datasets\wider_mafa_yolo`。
- 计划训练图片：WIDER train `12,880` + MAFA 难例 `3,000` = `15,880`。
- 主验证集：WIDER val `3,226` 张；MAFA test 保持独立，仅用于遮挡场景评估，不加入训练或主验证。

目录结构应类似：

```text
D:\face_train\datasets\wider_mafa_yolo\
  data.yaml
  images\
    train\
      wider\...
      mafa_3k\...
    val\
      wider\...
    mafa_test\...
  labels\
    train\
      wider\...
      mafa_3k\...
    val\
      wider\...
    mafa_test\...
```

`data.yaml`：

```yaml
path: D:/face_train/datasets/wider_mafa_yolo
train: images/train
val: images/val
names:
  0: face
```

关键规则：Ultralytics 会递归读取 `images/train` 子目录。若全量 MAFA 的 `images/train/mafa` 仍在训练路径下，即使另有 `mafa_3k`，也会把全量 MAFA 一并读入。应先将全量目录移出训练路径或移除该目录，再核对实际训练图片数为 `15,880`。不要把 MAFA test 放入 train/val。

## 2. 环境

本机已记录的训练环境可用（pip intall requirements-aims5702-cu128.txt导入）：

- Python：`D:\conda_envs\yolov11\python.exe`（Python 3.10 系列）
- PyTorch：`2.11.0+cu128`
- CUDA：可用
- GPU：NVIDIA GeForce RTX 5060，约 8 GB 显存
- Ultralytics：`8.4.193`

在 PowerShell 检查环境：

```powershell
& 'D:\conda_envs\yolov11\python.exe' -c "import torch, ultralytics; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('ultralytics', ultralytics.__version__)"
```

若 CUDA 不可用，不要开始长时间训练；先解决 PyTorch/CUDA 环境。依赖安装应按项目实际使用的 requirements 和受支持版本执行，避免训练过程中升级包。

## 3. 数据抽样与训练前检查

MAFA 难例抽样工具：`D:\face_train\sample_mafa_3k.py`。固定 seed 为 `42`，先 dry-run 检查源目录和可抽样数量，再执行：

```powershell
& 'D:\conda_envs\yolov11\python.exe' 'D:\face_train\sample_mafa_3k.py' --write
```

脚本参数以本机文件的 `--help` 输出为准。抽样后检查：

- WIDER train 图片与标签配对数为 `12,880`。
- MAFA 子集图片与标签配对数为 `3,000`。
- 全量 MAFA 目录不再位于 `images/train` 和 `labels/train` 的递归扫描路径中。
- WIDER val 为 `3,226` 张，标签配对完整。
- `data.yaml` 指向正确数据根目录。
- 训练集有效总图片数为 `15,880`，不是 `12,880 + 25,876`。

建议用文件管理器或脚本做成对计数，并确认标签与图像相对路径、文件名完全对应。检查随机图片和标签，确认 YOLO 标签为 `class x_center y_center width height` 的归一化格式，值在合法范围内。现有抽查记录：WIDER train `159,393` 个有效框、27 个无效框被跳过；WIDER val `39,697` 个有效框、11 个无效框被跳过；missing/unreadable 均为 0。数据若被重新转换，必须重新检查，不能沿用旧检查结论。

每 epoch 的步数可用来发现数据配置错误：`15,880 / batch 16` 约为 `993` 个 batch/epoch（取整方式由 dataloader 决定）。步数与该数量明显不符时，先查目录递归扫描、batch、缓存和数据集配置。

## 4. 启动训练

训练入口记录为 `D:\face_train\train_yolo11n_face.py`。先确认该脚本在当前工作区存在，并读取 `--help`/默认值；若脚本不存在，应先恢复正确版本，不要用未经核实的替代脚本开始实验。

已使用的启动命令示例：

```powershell
& 'D:\conda_envs\yolov11\python.exe' `
  'D:\face_train\train_yolo11n_face.py' `
  --batch 16 `
  --name yolo11n_face_b16
```

本实验计划为 `imgsz=640`、`epochs=100`、固定 `batch=16`。脚本应拒绝在 CUDA 不可用时训练，并在训练完成后导出 ONNX。训练从 `.pt` 开始，不能把 `.onnx` 当作 Ultralytics 常规微调权重。`cache`、Mosaic 等参数须按脚本与本次实验配置明确记录；比较实验时一次只改变有记录的变量。

恢复训练前先核对上次 run 的 `args.yaml`、日志、`results.csv`、`last.pt` 与 `best.pt`。明确指定 resume/checkpoint；不要通过重跑同一命令猜测它会从哪恢复。一次中断后，最终 epoch 数可能不是原计划数。

## 5. 训练监控与结果判读

常见日志字段：

- `box_loss`、`cls_loss`、`dfl_loss`：训练损失，观察趋势，不单独作为验收指标。
- `P`、`R`：Precision、Recall。
- `mAP50`、`mAP50-95`：WIDER val 上的检测指标。
- `steps/epoch`、`it/s`、验证耗时、checkpoint 写入和 ONNX 导出耗时：一起判断训练速度。

曾观察到约 `9.5–11 it/s` 的正常训练速度，也出现个别 epoch 降至约 `1 it/s`。GPU 显存或利用率单一数值不能解释原因；若速度异常，结合磁盘、CPU、其他 GPU 进程、dataloader 和系统内存观察。没有指标崩溃时，避免训练中途改变 cache/batch/Mosaic，以保持实验可比性。

本次已报告的一次训练结果（run：`D:\face_train\runs\yolo11n_face_b16_test-2`）：

- 日志称 `39 epochs completed in 4.381 hours`，虽最初计划 100 epochs；提前结束/恢复原因尚未查明。
- `best.pt` 最终验证：P `0.850`、R `0.615`、mAP50 `0.701`、mAP50-95 `0.370`。
- 验证集 `3,226` 张、`39,696` instances。
- RTX 5060 上报告 inference `0.8 ms/image`，该结果不是 RK3576 板端延迟。

此结果适合作为当前基线，不是通用达标结论。正式报告应记录实际 epoch、best epoch、数据路径/数量、评估阈值、软件版本与模型校验值。mAP50-95 与 Recall 的目标需根据产品对漏检/误检的要求定义；小脸、密集人脸和 MAFA 遮挡应分场景另行评估。

### FDDB 独立验证（RK3576 FP16 RKNN）

测试摘要来自 `/home/industio/yolo_test/yolo11n_face_fp16.rknn`，数据集为 `/home/industio/yolo_test/fddb_yolo` 的 `val` split。此结果是 RKNN 模型在 FDDB 上的独立评估，不是 WIDER val 训练指标，也不是 INT8 量化结果。

| 项目 | 结果 |
|---|---:|
| 图片数 | 567 |
| Ground-truth boxes | 1,054 |
| Confidence threshold | 0.25 |
| NMS IoU threshold | 0.45 |
| Precision/Recall 计数 IoU | 0.50 |
| TP / FP / FN | 988 / 120 / 66 |
| Precision | 0.8917 |
| Recall | 0.9374 |
| mAP50 | 0.9317 |
| mAP50-90 | 0.4444 |
| 检测总数 | 1,108 |
| 平均 confidence | 0.8050 |
| 推理延迟 mean / median / p95 | 51.765 / 51.874 / 58.176 ms/image |

AP 按 IoU 阈值：

| IoU | AP |
|---:|---:|
| 0.50 | 0.9317 |
| 0.55 | 0.9056 |
| 0.60 | 0.8396 |
| 0.65 | 0.6992 |
| 0.70 | 0.4578 |
| 0.75 | 0.1479 |
| 0.80 | 0.0168 |
| 0.85 | 0.0007 |
| 0.90 | 0.000016 |

解读：FDDB 上较高的 mAP50/Recall 与高 IoU AP 快速下降同时存在，说明模型能较多检出人脸，但定位框在严格 IoU 下与标注吻合度较弱。`mAP50-90` 是该摘要中跨 IoU 0.50–0.90 的指标，不等同于 Ultralytics 常报的 `mAP50-95`。51.8 ms/image 对应约 19.3 FPS（仅按该平均延迟倒数估算）；需核对计时范围是否只含 RKNN 推理、是否包含预处理/后处理，以及是否确为板端测量，不能直接与训练日志中的 GPU inference speed 比较。

`summary.json` 中模型命名为 `yolo11n_face_fp16.rknn`，因此本结果标记为 FP16 基线；它不能证明之后的 INT8 量化精度或速度。做 FP16/INT8 对比时必须使用同一数据、阈值、预处理、后处理与计时边界，并另存 INT8 结果。

## 6. 训练完成后的权重与 ONNX

优先使用本次 run 的：

```text
D:\face_train\runs\<run-name>\weights\best.pt
```

不要混用部署仓的旧模型：

```text
D:\yolo_rk3576_deploy\models\onnx\yolo11n.onnx
D:\yolo_rk3576_deploy\models\rknn\yolo11n_rk3576_fp16.rknn
D:\yolo_rk3576_deploy\models\rknn\yolo11n_rk3576_int8.rknn
```

如果训练脚本没有成功导出 ONNX，从新训练得到的 `best.pt` 导出；具体命令以当前 Ultralytics 版本和训练脚本为准，例如：

```powershell
& 'D:\conda_envs\yolov11\python.exe' -c "from ultralytics import YOLO; m=YOLO(r'D:\face_train\runs\<run-name>\weights\best.pt'); print(m.export(format='onnx', imgsz=640, dynamic=False, opset=12, simplify=True))"
```

导出后用 ONNX Runtime 检查实际输入输出的名称、shape 和 dtype，并做真实图像推理。单类 YOLO11n 常见输出可能是 `(1, 5, 8400)`，但必须读取实际 ONNX，不能依文件名或经验假定。YOLOv8/11 检测输出没有额外 objectness；后处理不要套用 YOLOv5 的 `4+1+nc` 布局，也不要重复 sigmoid。

## 7. RKNN 转换前后

- PC 端转换/校准使用 Linux 环境中的 RKNN Toolkit2；Windows 本机已知路径需重新核实后再使用。可在 WSL2 Ubuntu 或兼容 Linux x86_64 环境中配置匹配版本的 Python/wheel。
- `rknn_api`/`librknnrt` 是板端运行时，部署在 RK3576 Linux；与 PC 端 Toolkit2 不是同一个安装任务。
- 先做 FP16 RKNN 基线，再用代表性校准集做 INT8。
- 检测模型校准集使用完整场景图；识别 embedding 模型校准集使用人脸 crop，不可混用。
- 对检测后处理先检查 ONNX/RKNN 输出布局、dtype、scale/zero point、DFL 解码、stride 和 sigmoid 位置，再改 C++。
- 在 RK3576 上使用相同输入、阈值和后处理比较 FP16/INT8 的框、confidence、P/R、mAP 和实际板端延迟。

## 8. InsightFace 识别模型（独立任务）

InsightFace `w600k_mbf.onnx` 是 112×112 人脸识别 embedding 模型，不是 YOLO 检测器。当前文件位于 `D:\PTQ\models\w600k_mbf.onnx`，下载与 ONNX Runtime 冒烟测试脚本位于 `D:\PTQ\download_insightface_onnx.py`。已验证 I/O：输入 `input.1`，动态 batch、`3×112×112`、float；输出 `516`，`1×512`、float。预处理应按实际使用代码核实颜色顺序、对齐、crop、归一化和输出归一化，不能仅凭模型名假设。

WIDERFace 验证集的人脸框可用于生成识别模型 crop 校准数据：`D:\PTQ\build_calib_images.py`。该脚本默认读取 `images/val/wider` 与对应 YOLO 标签，默认抽 1,000 框、seed 42、扩边 20%、输出 112×112；需要注意 bbox 裁剪不等于关键点对齐。若实际部署有对齐步骤，calib 与 held-out 评估也应执行同样对齐。校准集与独立评估集应分开；通过 embedding cosine drift 与识别指标评估 FP32/INT8，不以模型“能转换”作为精度通过标准。

## 9. 复现记录清单

每次实验保存以下信息：

- 日期、代码版本/脚本哈希、数据集版本与目录清单。
- WIDER/MAFA 实际图片数、标签数、筛选 seed、抽查结果。
- Python、PyTorch、CUDA、GPU、Ultralytics 版本。
- 完整命令行、batch、imgsz、epochs、cache、数据增强和 resume 来源。
- run 目录、实际完成 epoch、best epoch、best.pt 路径。
- results.csv、验证指标、ONNX 路径与 I/O 信息。
- RKNN Toolkit2 版本、target、校准数据版本、FP16/INT8 对比与板端延迟。

复现原则：保留原始数据和已有 run；每次实验使用独立 run name；不覆盖用户模型；所有最终结论均以实际文件、日志与测试输出为依据。

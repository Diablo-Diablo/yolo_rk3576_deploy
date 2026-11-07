# YOLO11 on RK3576：端侧人脸检测与识别

面向触觉智能 Purple Pi OH2（Rockchip RK3576）的模型转换、PTQ 量化和板端推理项目。当前工作重点是 YOLO11n 人脸检测，以及后续接入 InsightFace MobileFaceNet 的分阶段身份识别。

人脸模型已完成静态 ONNX 导出、FP16/INT8 RKNN 转换及单输出模型板端排查。通过拆分框坐标和置信度输出，已在 PC 模拟器中消除 INT8 置信度全零现象；新 split RKNN 的板端效果、完整数据集精度和性能仍待验证。

## 环境

| 项目 | 配置 |
|---|---|
| 目标设备 | Purple Pi OH2 / RK3576，双核 NPU，标称 6 TOPS |
| 开发主机 | Windows + WSL，转换工作在 Linux 环境运行 |
| 项目路径 | Windows：`D:\yolo_rk3576_deploy`；WSL：`/mnt/d/yolo_rk3576_deploy` |
| 转换解释器 | `/opt/yolo_project/conda_envs/rknn_yolo/bin/python` |
| 转换工具 | RKNN Toolkit2 2.3.2 |
| 板端推理 | RKNN Runtime / RKNNLite；C++ 摄像头程序使用 V4L2 与 OpenCV |

## 当前进度

| 工作项 | 状态与验证范围 |
|---|---|
| YOLO11n 单类人脸模型 | 已有训练后 PT 和 ONNX；训练方案为 WIDER train 全量 + MAFA train 3,000 张难例，WIDER val 为主验证集 |
| 原单输出 FP16/INT8 RKNN | 已转换并进行板端同图诊断，INT8 原输出置信度全零，FP16 分数正常 |
| Split ONNX | 已生成 boxes/scores 双输出，ONNX checker 通过，同图数值与原 ONNX 对应通道完全一致 |
| Split INT8 PTQ | 使用 W8A8 + normal，未配置手工混合精度；已生成 RKNN，完成重新 build 后的 PC 模拟单图检查 |
| Split 板端与摄像头 | 待验证；原单输出程序需要适配双输出接口 |
| Split 数据集 benchmark | 脚本已编写，解析/NMS/指标函数做过局部检查，尚未运行真实数据集 benchmark |
| InsightFace MobileFaceNet | 已有 `w600k_mbf.onnx`；FP16/INT8 转换、识别效果和板端耗时待验证 |
| 语音唤醒与自动休眠 | 设计方向，尚未完成端到端集成验证 |
| 工业斗齿检测 | 后续方向，独立校准集及 mmse 量化仍为规划，不列为已完成成果 |

训练数据方案不等同于当前 checkpoint 的完整可复现训练记录；本仓库首页不引用未经核实的训练精度。

## 人脸检测 PTQ：问题与解决

原模型输出为 `(1,5,8400)`：4 个坐标通道与 1 个 face 分数通道。板端查询发现，该 INT8 输出共用 `scale=2.62702274`、`zero_point=-124`，分数通道的原始整数全部落在零点，反量化后为零。同图 FP16 最大置信度为 `0.806152344`。

当前方案移除最终输出 Concat，保留两个独立输出：

```text
YOLO11n → 解码后坐标 → boxes  (1,4,8400)
        → Sigmoid分数 → scores (1,1,8400)
```

这不增加张量维度，也不修改权重；它将不同数值范围的输出分开，避免坐标与分数共用粗粒度输出量化范围。转换仍是普通 INT8 PTQ，不是两份模型，也不要求手工混合量化。

使用原 1,000 张校准图片重新构建的真实 PC 模拟单图结果：

| 指标 | Split ONNX | Split INT8 模拟 |
|---|---:|---:|
| 最大置信度 | 0.8111753464 | 0.8370391726 |
| 分数 ≥ 0.25 的候选数（NMS 前） | 114 | 118 |

该结果来自 `wider_val_0000.jpg`，仅说明分数不再全零，不证明精度提升或板端性能达标。模拟器是从 ONNX 重新 build 后推理，不能直接对已导出 RKNN 使用 `load_rknn()` 做 PC 模拟。

曾尝试旧式非对称量化参数写法和 `kl_divergence` 算法。量化参数应传给 `rknn.config()`，当前版本使用已验证支持的 `w8a8`。KL 校准耗时较高，当前改回 `normal`；耗时应归因于具体校准算法和数据规模，不能简单归因于非对称量化。完整记录见 [YOLO11 人脸 PTQ](docs/YOLO11_FACE_PTQ.md)。

## 转换入口

以下命令在 WSL 中运行。拆分脚本与转换脚本都拒绝覆盖已有输出，重复实验需指定新路径。

```bash
# 从原单输出 ONNX 生成双输出 ONNX；已有 split 文件时不要重复执行。
/opt/yolo_project/conda_envs/rknn_yolo/bin/python \
  /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/split_face_onnx.py

# 普通 INT8 PTQ；如该目录已有同名模型，请更换 --output-dir。
/opt/yolo_project/conda_envs/rknn_yolo/bin/python \
  /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/rknn_yolo11_ptq.py \
  --onnx /mnt/d/yolo_rk3576_deploy/models/yolov11n/yolo_face/yolo11n_face_split.onnx \
  --mode int8 \
  --output-dir /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/output_split_normal
```

校准清单为 `scripts/YOLO11_face_PTQ/rknn_calib_1000/dataset.txt`，一行一张 WSL 可访问的图片路径。检测模型输入为 RGB uint8，RKNN 配置 mean=0、std=255，避免在主机侧重复归一化。本次校准使用了 WIDER val 的图片，后续精度评估应明确重叠情况并保留独立评估集。

## 后续人脸识别设计

采用按需、分阶段推理，避免持续对每帧的每张脸重复提取特征：

```text
触发采集 → YOLO检测 → 缓存合格的人脸ROI
                         ↓ 达到数量/质量条件或超时
                    停止YOLO推理
                         ↓
               筛选/去重/裁剪及必要的对齐
                         ↓
               MBF逐张embedding → L2归一化
                         ↓
               gallery余弦比对 → 身份/未知
```

`w600k_mbf.onnx` 输入为人脸 RGB 图，空间尺寸 112×112，输出为 `(1,512)` 特征向量，后处理需要 L2 归一化。计划预处理为 `(pixel-127.5)/128`，转换时需与实际使用的识别预处理核对。

缓存需要设数量上限并过滤低质量、重复图像；多人需按身份轨迹分组。当前 YOLO 只输出框，不提供五点关键点，后续需评估直接裁剪与对齐方案的识别效果。

“所有缓存人脸在 1,000 ms 内完成 embedding 和比对”是待测目标，受图片数量、预处理、模型初始化及 MBF 实际耗时影响，尚无板端数据支持保证。模型与 gallery 可考虑预加载，冷启动与热运行分别计时。

## 主要目录

```text
models/
├── yolov11n/
│   ├── yolo_face/                 # 人脸 PT、原 ONNX 和 split ONNX
│   └── rknn/                      # 板端模型资产
└── insightface/w600k_mbf.onnx     # 人脸 embedding 模型
scripts/YOLO11_face_PTQ/          # 校准、输出拆分、PTQ 和模拟检查
cpp_deploy/                      # 摄像头 C++ 与 CMake
benchmark_yolo_face/
├── benchmark_fp16/              # 原单输出评测入口
└── benchmark_int8_split/        # 双输出评测入口
docs/                           # 环境、部署、训练与 PTQ 记录
```

## 下一步验证

1. 核对本机/板端 split 模型哈希，查询两路输出属性并测试同图分数和框。
2. 适配摄像头 C++ 双输出读取，再运行小规模和完整数据集评估。
3. 统一 FP16/INT8 的数据、阈值、预处理和 AP 定义，分别记录精度与端到端耗时。
4. 建立 MBF 浮点识别基线，使用人脸裁剪校准集做 PTQ，检查同人/异人相似度分布。
5. 测量缓存整批人脸的处理延迟，确定采样数量和超时策略，再集成唤醒流程。

Benchmark 当前采用 YOLO 矩形标签与置信度截断后的 all-points AP，不是官方 FDDB 椭圆评测。PC 模拟耗时不作为 NPU 性能结论。

## 文档

- [人脸模型 PTQ 流程与问题记录](docs/YOLO11_FACE_PTQ.md)
- [训练复现记录](docs/TRAINING_REPRODUCTION.md)
- [板端 C++ 环境](docs/cpp_environment_setup.md)
- [摄像头记录](docs/web_cam_rk3576.md)
- [Purple Pi 设置与测试](docs/Purple_Pi_setup_test.md)
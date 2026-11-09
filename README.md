# YOLO11 on RK3576：端侧人脸检测与识别

面向触觉智能 Purple Pi OH2（Rockchip RK3576）的模型转换、PTQ 量化和板端推理项目。人脸部分采用 YOLO11n split INT8 检测、InsightFace MobileFaceNet FP16 特征提取和 reference 余弦比对。

**人脸检测与分阶段识别已完成原型阶段验收（用户确认）。** 已完成 split INT8 板端检测、FDDB 验证集评测、MBF FP16 C++ 集成，以及本人/他人摄像头采集测试。验收范围为逐 ROI 匹配和已测批量延迟；独立会话统计、活体检测与门锁控制仍属于后续工作。

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
| Split 板端与摄像头 | 已适配双输出，并完成板端相机检测与识别链路运行 |
| Split 数据集 benchmark | 已评测 FDDB val 567 张图片、1,054 个 GT；AP50 90.35%，Precision 85.92%，Recall 91.46% |
| InsightFace MobileFaceNet | FP16 RKNN 已接入 C++，输出512维归一化embedding；MBF INT8不作为当前方案 |
| YOLO + MBF 集成 | 原型阶段验收完成；run4 本人14/17个ROI通过，run5他人0/15通过；整批识别分别673.23/699.01 ms |
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

## 已实现的人脸识别管线

采用按需、分阶段推理，避免持续对每帧的每张脸重复提取特征：

```text
触发采集 → YOLO检测 → 缓存合格的人脸ROI
                         ↓ 达到数量/质量条件或超时
                    停止YOLO推理
                         ↓
               缓存ROI逐张缩放至112×112（当前无对齐）
                         ↓
               MBF逐张embedding → L2归一化
                         ↓
               reference余弦比对 → 逐ROI match
```

板端入口为 [face_recognition.cpp](cpp_deploy/face_recognition.cpp)，CMake target 为 `face_recognition`。MBF 使用 `w600k_mbf_rk3576_fp16.rknn`，输入 RGB 112×112，输出 `(1,512)`，主机后处理执行 L2 归一化。RKNN 内编码 `(pixel-127.5)/128`，C++ 输入 UINT8，不重复归一化；动态输入模型初始化时显式设置输入形状。

Reference 统一位于 `face_recog/embedding_reference/owner.npy`，板端对应 `/home/industio/yolo_test/embedding_reference/owner.npy`。录入入口 `scripts/face_recog_pipline/Yolo_insightface_integration/enroll.py` 默认写入此目录；原 `face_recog/gallery/owner.npy` 保留。当前只做逐ROI匹配，未实现跟踪、去重、五点对齐、活体或整次解锁判定。

已测 run2/run4/run5 分别缓存15/17/15个ROI，整批识别耗时619.43/673.23/699.01 ms，均低于1,000 ms。该计时包含逐ROI embedding、余弦比对、PNG保存及循环内报告写入，不含采集、相机预热、模型加载和最终报告/NPY写入；不代表50个ROI或跨会话P95已达标。

### 已确认的验收口径

用户确认当前人脸原型可完成验收，并采用以下参数口径。参数确认与统计验证状态分开记录；本次只更新文档，不修改程序默认值。

| 项目 | 确认口径 | 当前证据/状态 |
|---|---|---|
| 识别阈值 | cosine ≥ 0.43，固定当前owner reference | run4本人多数ROI通过；用户确认run5为他人且全部拒绝 |
| 检测参数 | conf=0.25，NMS IoU=0.45 | 已测运行基线 |
| 采集 | 3秒、5次/秒，目标15帧 | 相机预热约2秒另计 |
| 批量预算 | 原型验收以≤15个合格ROI、整批识别≤1,000 ms为目标 | 已测15～17个ROI通过；代码max-faces仍为50 |
| ROI裁剪基线 | 最小边40像素，margin=0.2 | 已测；后续质量增强建议最小边60、宽高比0.6～1.5，尚未编码或复验 |
| 延迟扩展验收 | 至少30次独立会话，P95≤1,000 ms | 已认可为后续验证标准，尚无该规模数据 |
| 身份扩展验收 | 本人至少20次、至少5位他人各10次独立触发；本人会话通过率≥95%，该轮他人零放行 | 后续验证标准；当前两次采集不能证明总体FAR/FRR |
| 会话判定与兜底 | 后续单人模式至少3个合格ROI且至少3个通过；多人/不足/异常拒绝或重采 | 尚未实现，当前验收止于逐ROI输出，无门锁动作 |

当前原型验收依据和参数详见 [人脸管线总结](docs/yolo_face_SUMMARY.md)，原始证据位于 [集成结果归档](benchmark_yolo_face/yolo_mbf_integrate/)。连续帧的ROI通过比例不等于独立会话准确率，run5零通过不等于总体误接受率为零。

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
├── benchmark_int8_split/        # 双输出评测入口及results_board实测
└── yolo_mbf_integrate/          # run2/run4/run5、单图测试原始结果与汇总
face_recog/embedding_reference/  # 统一reference目录
scripts/InsightPTQ/              # MBF FP16转换及产物
scripts/face_recog_pipline/Yolo_insightface_integration/
                                # PC原型、录入和相机检查
docs/                           # 环境、部署、训练与 PTQ 记录
```

## 后续工作（不阻塞当前原型验收）

1. 按已确认标准扩充独立会话与不同身份测试，统计P95和会话通过/拒绝情况。
2. 实现并复验ROI质量过滤与单人会话判定；保持模型/reference/阈值版本可追溯。
3. 统一FP16/INT8评测的NMS与匹配规则后再严格比较量化损失。
4. 集成语音唤醒、错误兜底与按需启动；如需门禁用途，再评估活体与攻击防护。
5. 工业斗齿使用独立数据和校准流程，不将本次人脸验收扩展为工业任务验收。

Benchmark 当前采用 YOLO 矩形标签与置信度截断后的 all-points AP，不是官方 FDDB 椭圆评测。PC 模拟耗时不作为 NPU 性能结论。

## 文档

- [人脸模型 PTQ 流程与问题记录](docs/YOLO11_FACE_PTQ.md)
- [人脸集成总结与验收口径](docs/yolo_face_SUMMARY.md)
- [板端集成测试汇总CSV](benchmark_yolo_face/yolo_mbf_integrate/run_summary.csv)
- [相关文件夹树](benchmark_yolo_face/yolo_mbf_integrate/related_tree.txt)
- [训练复现记录](docs/TRAINING_REPRODUCTION.md)
- [板端 C++ 环境](docs/cpp_environment_setup.md)
- [摄像头记录](docs/web_cam_rk3576.md)
- [Purple Pi 设置与测试](docs/Purple_Pi_setup_test.md)
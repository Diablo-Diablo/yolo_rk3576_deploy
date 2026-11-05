# YOLO11 人脸检测 PTQ 流程与问题记录

## 1. 目标与当前结论

将单类人脸 YOLO11n 从 PyTorch 导出为静态 ONNX，再通过 RKNN Toolkit2 2.3.2 转换为 RK3576 的 INT8 RKNN。

原单输出 INT8 模型在板端出现置信度全零。实测最终输出张量的量化步长为 2.62702274，框坐标与 0～1 的置信度共用输出量化参数，导致置信度落在量化零点。将最终输出拆成 boxes 和 scores 后，使用普通 W8A8、normal 算法重新量化，PC 模拟推理中置信度恢复非零。

当前结果仅覆盖单图检查。新 split RKNN 尚未完成板端验证及完整数据集精度评估，不能据此声称精度或性能达标。

## 2. 环境与文件

- Windows 项目：`D:\yolo_rk3576_deploy`。
- WSL 项目：`/mnt/d/yolo_rk3576_deploy`。
- 转换解释器：`/opt/yolo_project/conda_envs/rknn_yolo/bin/python`。
- 工具版本：RKNN Toolkit2 2.3.2；目标芯片：RK3576。
- 下列命令均在 WSL 终端执行。

以下路径相对于项目根目录：

| 文件 | 用途 |
|---|---|
| `models/yolov11n/yolo_face/yolo11n_face.onnx` | 原单输出 ONNX |
| `models/yolov11n/yolo_face/yolo11n_face_split.onnx` | 拆分输出后的 ONNX |
| `scripts/YOLO11_face_PTQ/split_face_onnx.py` | 拆分最终 Concat 输出 |
| `scripts/YOLO11_face_PTQ/select_wider_val_calib.py` | 生成校准图片及清单 |
| `scripts/YOLO11_face_PTQ/rknn_calib_1000/dataset.txt` | 1000 张校准图片的路径清单 |
| `scripts/YOLO11_face_PTQ/rknn_yolo11_ptq.py` | FP16/INT8 转换入口 |
| `scripts/YOLO11_face_PTQ/output_split_normal/yolo11n_face_split_rk3576_int8.rknn` | 当前找到的 split INT8 文件 |
| `scripts/YOLO11_face_PTQ/check_split_simulator.py` | 从 ONNX 重建量化模型并单图模拟 |
| `scripts/YOLO11_face_PTQ/split_simulator_result.json` | 真实 PC 模拟推理结果 |
| `scripts/diagnose_face_output.cpp` | 原单输出模型板端属性及输出诊断 |
| `benchmark_yolo_face/benchmark_int8_split/benchmark_fddb_rknn_split.py` | split 模型数据集 benchmark 入口 |

## 3. 实际流程

### 3.1 PT → 静态 ONNX

训练使用单类 face YOLO11n，导出时使用固定 640×640 输入、关闭动态形状。原 ONNX 的输出为 `(1, 5, 8400)`，其中 5 = 4 个框坐标 + 1 个人脸分数，无独立 objectness 通道。

先保留原 ONNX 和 FP16 RKNN 作为对照，不覆盖基线文件。

### 3.2 准备校准数据

本次使用 WIDER FACE 验证图片中的 1000 张作为校准输入，清单一行一张图片的 WSL 绝对路径，例如 `/mnt/d/.../image.jpg`。

转换脚本检查清单非空，并逐项检查文件存在。文件存在不等于图片可解码或具有代表性，后续仍应检查图片内容、尺寸和部署场景覆盖。

本次校准使用了验证图片，因此这部分图片上的结果仅用于调试；用于最终精度结论时，应明确校准/评估重叠，优先采用独立评估集。

预处理配置：

```python
mean_values = [[0, 0, 0]]
std_values = [[255, 255, 255]]
target_platform = "rk3576"
```

推理端输入 RGB uint8、范围 0～255，由模型内置预处理完成除以 255；不要在主机侧重复归一化。本次单图诊断使用直接 resize 到 640×640，并按 x/y 比例映射回原图。后续比较必须保持同一预处理策略。

### 3.3 原单输出模型转换与板端排查

先转换 FP16 和 INT8，摄像头 INT8 测试没有检测框。随后脱离摄像头，用相同图片、相同 RGB uint8 预处理运行两个模型。

板端与本机模型 SHA256 已核对一致，排除了当时板端模型与本机指定文件不同的情况。

原单输出板端实测：

| 项目 | INT8 | FP16 |
|---|---:|---:|
| 输出形状 | `(1,5,8400)` | `(1,5,8400)` |
| 输出量化 scale | 2.62702274 | 不以此解释浮点输出 |
| INT8 zero point | -124 | — |
| 分数通道原始整数值 | 全为 -124 | — |
| 反量化最大置信度 | 0 | 0.806152344 |
| 分数 ≥ 0.25 的候选数，NMS 前 | 0 | 114 |

初始化、输入设置、运行及输出获取均返回成功。问题发生在模型输出数值，不能通过降低 camera 阈值解决。

### 3.4 尝试非对称量化参数与 KL 算法

排查过程中尝试过以下配置写法：

```python
quantized_dtype = "asymmetric_affine-u8"
quantized_algorithm = "kl_divergence"
```

最初误将它们传给 `rknn.build()`，出现：

```text
RKNN.build() got an unexpected keyword argument 'quantized_dtype'
```

核对当前安装的 2.3.2 API 后，将量化配置移至 `rknn.config()`，dtype 改为该版本文档支持的 `w8a8`。`config()` 实测接受 `w8a8 + kl_divergence`，返回 0。

需区分两个概念：非对称量化描述 scale/zero point 的映射；KL divergence 是搜索量化阈值的校准算法。旧字符串 `asymmetric_affine-u8` 没有在本次实验中成功完成转换，不能把它直接当作当前版本 `w8a8` 的同义名称。原 INT8 输出的非零 zero point 则直接证明其输出使用了带偏移的仿射量化。

KL 阶段出现日志：

```text
Quantizating: 1/236 [00:57<3:46:43, 57.89s/it]
```

这是首个进度项的耗时及动态剩余时间估计，不是完整实测总耗时，在第一轮内存占用就爆了，WSL停止运行。结合 1000 张校准图片，其计算开销不适合当时快速排查，最终没有采用 KL 作为当前方案，回到 `normal`。

准确表述是：尝试过非对称量化相关配置和 KL 算法；当前放弃的是耗时的 KL 路线，不能把慢的原因简单归结为“非对称量化”。本次未证明 KL 能修复置信度全零。

### 3.5 拆分最终输出

原 ONNX 末端：

```text
解码后的 boxes (1,4,8400) ─┐
                          Concat → output0 (1,5,8400)
Sigmoid 后 scores (1,1,8400)┘
```

最终节点为 `/model.23/Concat_3`，两路输入分别来自 `/model.23/Mul_2_output_0` 与 `/model.23/Sigmoid_output_0`。

执行拆分脚本：

```bash
/opt/yolo_project/conda_envs/rknn_yolo/bin/python \
  /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/split_face_onnx.py
```

脚本暴露这两路张量，并用 Identity 给输出命名：

- `boxes: (1,4,8400)`，坐标含义为 cx、cy、w、h。
- `scores: (1,1,8400)`，已经过 Sigmoid，不要再次调用 Sigmoid。

这是两个三维输出张量，不是升维。权重不变，不需要重新训练。脚本拒绝覆盖已有目标文件；重复生成时用 `--output` 指定新路径。

新旧 ONNX 已用 ONNX Runtime 做同图对比：boxes、scores 与原输出对应通道的最大绝对误差均为 0。ONNX checker 通过。

### 3.6 Split ONNX → 普通 INT8 RKNN

当前 PTQ 默认输入已改成 split ONNX，使用普通量化：

```python
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform="rk3576",
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
)
rknn.load_onnx(model=".../yolo11n_face_split.onnx")
rknn.build(do_quantization=True, dataset=".../dataset.txt")
```

没有调用混合量化 step1/step2，也没有手工指定检测头保留 FP16。“分开量化”指在同一模型中保留两个独立输出张量，使坐标和分数能够使用独立的输出量化参数；不是生成两份模型，也不是仅量化最后两个输出。

可复现命令：

```bash
/opt/yolo_project/conda_envs/rknn_yolo/bin/python \
  /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/rknn_yolo11_ptq.py \
  --onnx /mnt/d/yolo_rk3576_deploy/models/yolov11n/yolo_face/yolo11n_face_split.onnx \
  --mode int8 \
  --output-dir /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/output_split_normal
```

当前同名输出已存在，脚本会拒绝覆盖；再次实验应指定新的输出目录。输出名为 `yolo11n_face_split_rk3576_int8.rknn`。

## 4. 根因分析

反量化公式：

```text
real_value = (quantized_value - zero_point) × scale
```

原模型的最终输出把数百量级的坐标和 0～1 的分数合并，输出共享 scale=2.62702274。分数通道原始整数全部为 -124，恰好等于 zero point，因此反量化后全为零。

证据强烈支持最终混合数值范围共用量化参数是主要原因。Concat 运算本身没有算错；问题在拼接后张量的量化分辨率。`want_float=1` 仅反量化，不能恢复已舍入丢失的分数。

拆分输出避免了这一共享输出范围，但不能保证内部所有节点均无量化误差，仍需板端和数据集验证。

## 5. 无板卡时的模拟检查

直接执行 `load_rknn()` 后 `init_runtime()` 无法在 PC 模拟器推理，Toolkit2 返回：

```text
RKNN model that loaded by 'load_rknn' not support inference on the simulator
```

可行路径是从 ONNX 按相同配置重新 `build()`，随后不指定 target 初始化模拟器。`check_split_simulator.py` 采用这条路径，使用原 1000 张校准图片，不覆盖已导出的 RKNN 文件。

```bash
/opt/yolo_project/conda_envs/rknn_yolo/bin/python \
  /mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/check_split_simulator.py
```

注意此命令会重新进行完整校准量化，即使最后只推理一张图片，也有构建耗时。

真实模拟结果，输入 `wider_val_0000.jpg`：

| 指标 | Split ONNX | Split INT8 PC 模拟 |
|---|---:|---:|
| 最大置信度 | 0.8111753464 | 0.8370391726 |
| 非零分数数量 | 6501 | 318 |
| 分数 ≥ 0.25 的候选数，NMS 前 | 114 | 118 |

分数不再全零；低分项数量因量化而变化。最大分数更高不等于精度提升。单图候选数接近不等于框位置、召回或 mAP 已验证。

这组结果来自重新构建的内存模型模拟推理，不是对已有 `.rknn` 文件直接执行，也不是板端 NPU 实测。

## 6. 踩坑与解决办法

| 问题 | 定位与解决 |
|---|---|
| 摄像头 INT8 没框 | 先同图比较 FP16/INT8，检查输出形状、类型、scale、zp、原始和反量化分数 |
| 怀疑 dataset.txt | 检查 WSL 路径、文件存在和图片可解码性；本次恢复分数使用同一校准清单，证据更支持输出量化范围问题 |
| 把量化参数放进 build | 改放 config；以安装版本的 API 签名和文档为准 |
| 直接沿用旧 dtype 字符串 | 2.3.2 当前采用已验证接受的 w8a8，不假设旧别名兼容 |
| KL 太慢 | 不采用其作为当前方案，切回 normal；首项 ETA 不作为总耗时结论 |
| 单输出坐标/分数共用 scale | 拆分 boxes/scores 后重新量化 |
| 输出转 float 仍全零 | 反量化不能恢复丢失信息，必须改量化边界/输出结构 |
| 输出数量变化 | 推理端获取两个输出，按形状或查询属性识别，不能继续按单输出 5 通道索引 |
| PC 直接加载 RKNN 模拟失败 | 从 ONNX load/build 后模拟；已有 RKNN 的实际行为留待板端确认 |
| Netron 查看 INT8 | 当前查看的是 split ONNX 结构，不包含 RKNN 输出 scale 等实际量化属性 |

## 7. 板端接入与剩余验证

新模型输出接口已变化，单输出 camera 程序不能直接复用原索引。板端需要：

1. 核对本机/板端 split RKNN 哈希。
2. 查询两个输出属性及各自 scale、zero point，确认哪个是 boxes、哪个是 scores。
3. C++ 获取两个输出；可用 `want_float=1` 分别反量化，再读取坐标和置信度。
4. 用相同图片比较 FP16 和 split INT8 的最大分数、NMS 后框位置及数量。
5. 小规模检查通过后，再运行完整 benchmark，随后接入摄像头。

Benchmark 脚本已写入 `benchmark_yolo_face/benchmark_int8_split/benchmark_fddb_rknn_split.py`，双输出解析及 NMS/指标函数经过局部 ad-hoc 检查，但尚未运行真实数据集 benchmark。

脚本采用 YOLO 矩形标签、置信度截断后的 all-points AP，不能称为官方 FDDB 椭圆评测。其推理计时包含 RKNN Python API 输入/输出处理，不等同于纯 NPU `rknn_run` 耗时。与现有 FP16 benchmark 比较时还应对齐阈值、预处理、数据划分及 AP 的 IoU 范围，避免直接比较不同定义的汇总指标。

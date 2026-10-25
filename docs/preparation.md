### 1. 异构开发环境搭建与工程隔离
- **环境架构解耦**：
  - 避免 Windows 宿主机与 WSL2 的 NTFS 文件系统 (`/mnt/d/`) 系统调用不兼容问题（`[Errno 95] Operation not supported / forkserver`），将 Conda 解释器及底层虚拟环境隔离部署在 WSL 本地 Linux 文件系统 (`/opt/yolo_project/conda_envs/rknn_yolo`)。
  - 项目工程资产、权重与代码统一映射在 `D:\` 盘，兼顾 Linux 工具链稳定编译与 Windows 宿主机高效开发协同。
- **RKNN 工具链与依赖冲突攻关**：
  - 锁定 Python 3.10 运行时基线。
  - 攻克官方 PyPI 绑死 CUDA 12 庞大依赖包下载过慢的问题，采用清华与阿里国内高速镜像打通全链路。
  - 修复 `setuptools >= 70.0.0` 移除 `pkg_resources` 导致的 RKNN 底层解析崩溃问题，版本回退并锁死在安全范围。
  - 修复 `onnx >= 1.16.0` 移除 `onnx.mapping` 导致的属性缺失报错，版本锁定在黄金兼容版本 `onnx==1.15.0`。
  - 验证成功导入 `rknn.api.RKNN`，完成端到端工具链本地自检。

### 2. 算子图优化与量化流验证
- **图优化流水线落地**：
  - 开启图调试模式验证了 RKNN 内部完整的 4 阶段算子图优化：
    1. `check0_base_optimize.onnx`（基础图优化与冗余消除）
    2. `check1_fold_constant.onnx`（常量折叠）
    3. `check2_correct_ops.onnx`（算子规整与 NPU 指令对齐）
    4. `check3_fuse_ops.onnx`（关键算子融合：Conv + BatchNorm + SiLU 深度融合）
  - 验证 YOLOv11n 结构 100% 兼容瑞芯微 RK3576 NPU 编译流，无不支持算子阻碍。

### 3. 模型输出头拓扑协议拆解 (Shape Protocol Dissection)
通过 x86 仿真环境直接探测输出张量，确定了端侧模型输出接口定义：
- **输出张量**：`outputs[0].shape = (1, 84, 8400)`，`dtype = float32`
- **维度解析**：
  - `1`: Batch size。
  - `8400`: 三个尺度特征图的候选 Anchor 总和（$80\times80=6400$ 抓小目标 + $40\times40=1600$ 抓中目标 + $20\times20=400$ 抓大目标）。
  - `84`: 前 4 个维度为已完成 DFL 解码的边界框 `[cx, cy, w, h]`；后 80 个维度为类别置信度。
- **量化机制确认**：
  - NPU 内部全算子走 **INT8 整数计算**以跑满 6 TOPS 硬件峰值算力。
  - 最后一层由驱动硬件快速反量化输出为连续 `float32`，板端免去手动反量化运算，后处理仅需标准的 **置信度过滤 + NMS**。

### 4. 双场景解耦量化策略（PTQ 落地规划）
- **人脸检测解锁（任务 A）**：采用 `normal` 非对称量化，聚焦暗光与面部轮廓连续性。
- **工业斗齿监测（任务 B）**：针对金属反光与泥沙遮挡下的极小目标，采用 `mmse`（最小均方误差）搜寻最优截断截点，避免微弱边缘信息被舍入吃掉。
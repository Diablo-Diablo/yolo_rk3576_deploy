# Edge AI Deployment: YOLOv11 on Rockchip RK3576 (Purple Pi OH2)

本项目致力于在端侧受限嵌入式硬件（**触觉智能 Purple Pi OH2 / Rockchip RK3576**，双核 NPU 算力 6 TOPS）上，实现 **YOLOv11** 目标检测模型的全流程工程化部署与量化落地。

项目涵盖两个核心业务场景：
1. **任务 A：端侧低功耗人脸识别解锁系统**（Porcupine 离线语音唤醒词触发 -> 按需拉起 YOLOv11n 推理 -> 自动休眠释放资源）。
2. **任务 B：重型工程机械视觉监测系统**（挖掘机斗齿实时检测与缺失/掉落预警，面向工业恶劣环境的小目标高可靠性检测）。

---

## 📌 硬件与系统架构

- **目标平台 (Target)**: Purple Pi OH2 (Rockchip RK3576J / Dual-Core NPU @ 6 TOPS)
- **开发与交叉转换主机 (Host)**: Windows 11 + WSL2 (Ubuntu 22.04 LTS x86_64)
- **核心推理栈**:
  - 主机端转换工具链: `rknn-toolkit2` v2.3.2
  - 板端推理轻量引擎: `rknpu2` / `rknn-toolkit-lite2`
  - 前处理加速: V4L2 视频采集 + 瑞芯微 RGA 2D 硬件图像加速引擎 (零拷贝/硬件缩放/色彩空间转换)
  - 模型架构: YOLOv11n (Exported via PyTorch with ONNX Opset 12)

---

## 🚀 当前工程进展 (Milestones & Work Done)

### 1. 异构开发环境搭建与工程隔离
- **解耦式环境架构**：
  - 避免 Windows 宿主机与 WSL2 的 NTFS 文件系统 (`/mnt/d/`) 系统调用不兼容问题（`[Errno 95] Operation not supported / forkserver`），将 Conda 解释器及底层虚拟环境隔离部署在 WSL 本地 Linux 文件系统 (`/opt/yolo_project/conda_envs/rknn_yolo`)。
  - 项目工程资产、权重与代码统一映射在 `D:\` 盘，兼顾 Linux 工具链稳定编译与 Windows 宿主机高效开发协同。
- **RKNN 工具链与依赖冲突攻关**：
  - 锁定 Python 3.10 运行时基线。
  - 攻克官方 PyPI 绑死 CUDA 12 庞大依赖包下载过慢的问题，采用清华与阿里国内高速镜像打通全链路。
  - 修复 `setuptools >= 70.0.0` 移除 `pkg_resources` 导致的 RKNN 底层解析崩溃问题，版本回退并锁死在安全范围。
  - 验证成功导入 `rknn.api.RKNN`，完成端到端工具链本地自检。

### 2. 模型资产准备与多场景量化管线设计
- **基线模型就绪**：
  - 成功获取并验证 YOLOv11n 浮点基线（PyTorch 权重转 ONNX Opset 12，完成算子化简与图优化）。
- **双场景解耦量化策略（PTQ）**：
  - **人脸检测解锁**：采用 `normal` 对称/非对称量化，聚焦暗光与面部五官梯度的稳定性。
  - **工业斗齿监测**：针对金属反光、泥沙遮挡及极小目标漏检痛点，量化算法规划采用 `mmse`（最小均方误差）优化截断点，设立独立的真实场景校准数据集 (`dataset_industrial.txt`)。

---

## 📂 项目工程目录规划

```text
.
├── calibration_data/              # 量化校准数据集 (解耦设计)
│   ├── face/                      # 人脸场景多光照/多角度校准图
│   │   └── dataset_face.txt
│   └── industrial/                # 挖掘机/工程机械实拍校准图
│       └── dataset_industrial.txt
├── models/                        # 模型资产库
│   ├── onnx/                      # 优化后的 ONNX 浮点模型
│   │   └── yolo11n.onnx
│   └── rknn/                      # 导出针对 RK3576 的 NPU 二进制模型
│       ├── yolo11n_face_rk3576_int8.rknn
│       └── yolo11n_teeth_rk3576_int8.rknn
├── convert/                       # 主机端交叉编译与量化脚本
│   ├── convert_yolo11.py          # ONNX -> RKNN 转换与 INT8 量化主程序
│   └── verify_simulation.py       # x86 主机端 RKNN 仿真验证脚本
├── board/                         # 板端 C++ / Python 推理与部署代码
│   ├── include/                   # 瑞芯微 rknn_api / rga 头文件
│   ├── src/                       # 后处理 (DFL 解码、NMS) 与硬件拉流代码
│   └── test_rknn_board.py         # 板端快速联调原型脚本
└── docs/                          # 实验记录与部署技术文档

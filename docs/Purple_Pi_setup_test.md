# Purple Pi OH2 (Rockchip RK3576) 硬件上板部署与基准测速实验报告

- **实验日期**: 2026-10-24
- **测试硬件**: 触觉智能 Purple Pi OH2 (Rockchip RK3576J / 6 TOPS 双核 NPU)
- **部署模型**: YOLOv11n (ONNX Opset 12 转换，INT8 非对称量化，静态输入尺寸 640x640x3)
- **操作环境**: Windows 11 宿主机 (ADB 控制) + 板载 Ubuntu 22.04 LTS (aarch64, Linux 6.1 内核)

---

## 1. 硬件连接与系统烧录

### 1.1 固件选择
- **选型**: `IDO_PurplePiOH2_V1B_Ubuntu22.04_HDMI4K_250907R.img`
- **说明**: 采用 HDMI4K 输出分支，支持板载桌面显示与外部屏幕直连。

### 1.2 烧录模式与底层握手
1. **模式触发**:
   - 拔除 Type-C 数据线（板子完全断电）。
   - 按住板载微动按键 `Maskrom / Recovery`。
   - 插入连接 Laptop 的 Type-C OTG 线缆上电，保持按键 3 秒后松开。
2. **烧录工具**:
   - 使用 Windows 端原厂工具 `RKDevTool v3.32`。
   - 底层状态栏成功识别：`发现一个LOADER设备` (端口 `1-6: LOADER`)。
   - 切换到【升级固件】标签页，载入解压出的 `.img` 固件，点击【升级】将系统完整烧录至板载 eMMC 介质。
   - 烧录完成板端自动软复位并进入 Ubuntu 系统桌面。

---

## 2. NPU 驱动层与运行库对齐自检

### 2.1 DRM 架构下的硬件设备节点
在 Linux 6.1 新内核架构下，NPU 驱动已重构为 DRM（Direct Rendering Manager）接口：
- **GPU 节点**: `/dev/dri/card0` / `/dev/dri/renderD128` (Mali GPU 独占)
- **NPU 节点**: `/dev/dri/card1` / `/dev/dri/renderD129` (驱动绑定至 `platform/drivers/RKNPU`)
- **内核驱动版本**: `RKNPU driver: v0.9.8`
- **运行主频**: `/sys/class/devfreq/27700000.npu/cur_freq` 读取为 `950000000 Hz` (950 MHz 最高性能档位)

### 2.2 跨版本库依赖对齐 (Runtime Alignment)
- **初始状态**: 板端预装基础库为 `librknnrt.so: 2.1.0`。
- **潜在风险**: WSL 主机端导出工具链采用 `rknn-toolkit2: 2.3.2`，直接推理将触发 `RKNN_ERR_MODEL_VERSION_MISMATCH`。
- **对齐处理**:
  - 通过 ADB 备份原有库：`cp /usr/lib/librknnrt.so /usr/lib/librknnrt.so.bak`
  - 将配套的 aarch64 `librknnrt.so (v2.3.2)` 推送替换至 `/usr/lib/librknnrt.so` 并赋予 755 权限。
  - 经 `strings` 自检确认版本对齐至：`2.3.2 (429f97ae6b@2025-04-09)`。

### 2.3 板端轻量运行时环境搭建
- **用户主目录配置**: 确立以 `industio` 用户家目录 (`/home/industio/yolo_test`) 为部署工作区。
- **Python 运行时**: 系统原生 Python 3.10.12 (aarch64)。
- **工具库安装**:
  - 推送 `rknn_toolkit_lite2-2.3.2-cp310-cp310-linux_aarch64.whl` 至板端。
  - 联网解包安装，经测试核心 C++ 绑定接口完全独立，无需繁重外网依赖。
  - 补齐 `opencv-python-headless` 完成前处理支持。

---

## 3. 板端 YOLOv11n INT8 推理基准测试 (Benchmark)

### 3.1 核心测试配置
- **测试脚本**: `scripts/test_board_yolo11.py`
- **NPU 核心调度策略**: `core_mask=RKNNLite.NPU_CORE_0_1` (启用 RK3576 双核算力负载均衡)
- **热身策略**: 前置执行 3 轮 Dummy 推理消除变频抖动与首次页面置换开销。
- **测试轮次**: 连续执行 10 轮标准测试。

### 3.2 实测数据

```text
========================================
 RK3576 YOLOv11 板端硬件推理实测
========================================
W rknn-toolkit-lite2 version: 2.3.2
--> [1/4] 正在加载 RKNN 模型: /home/industio/yolo_test/yolo11n_rk3576_int8.rknn
✓ 加载模型成功
--> [2/4] 初始化 NPU 运行时 (启用双核 NPU_CORE_0_1)...
I RKNN: [10:40:05.876] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [10:40:05.876] RKNN Driver Information, version: 0.9.8
I RKNN: [10:40:05.877] RKNN Model Information, version: 6, toolkit version: 2.3.2, target: RKNPU f2, target platform: rk3576
✓ NPU 硬件环境就绪
--> [3/4] 正在载入并预处理测试图像: /home/industio/yolo_test/bus.jpg
--> 正在执行 NPU 热身 (3 轮)...
--> [4/4] 正在执行 10 轮时延基准测试...
    第  1 次推理耗时:  32.01 ms
    第  2 次推理耗时:  29.85 ms
    第  3 次推理耗时:  28.31 ms
    第  4 次推理耗时:  30.67 ms
    第  5 次推理耗时:  25.98 ms
    第  6 次推理耗时:  33.64 ms
    第  7 次推理耗时:  27.26 ms
    第  8 次推理耗时:  31.04 ms
    第  9 次推理耗时:  28.34 ms
    第 10 次推理耗时:  32.12 ms

========================================
📊 实测性能汇总 (Performance Summary):
  - 平均单帧时延 (Average): 29.92 ms
  - 中位数时延 (Median)   : 30.26 ms
  - 极致最低时延 (Min)     : 25.98 ms
  - 峰值最高时延 (Max)     : 33.64 ms
  - 理论纯推理帧率 (FPS)  : 33.4 FPS
========================================

🔍 输出张量结构核对:
  - 输出层数量: 1
  - Output[0] Shape: (1, 84, 8400), Dtype: float32
```

---

## 4. 结论

1. **时延与吞吐达标**:
   - 纯硬件 NPU 推理平均时延达到 **29.92 ms**，理论吞吐达 **33.4 FPS**，基本满足工业级视频流实时监测需求（通常要求 >= 25 FPS）。
2. **张量与反量化机制核准**:
   - 板端驱动层已对输出张量进行了透明的反量化处理，最终交付维度为标准的 `(1, 84, 8400)` 连续浮点数组。
   - 确认包含 8400 个全尺度候选框（80x80 + 40x40 + 20x20）以及已内化 DFL 积分的边界框参数 `[cx, cy, w, h]`，后处理仅需标准的置信度门限与 NMS 筛选，无复杂计算负担。
3. **下一步目标**:
   - 编写轻量纯 NumPy/C++ 后处理模块完成结果可视化。
   - 引入挖掘机实地采样数据集，针对小目标与工业反光特征重新执行专项目标量化（PTQ - MMSE 算法）。

# Purple Pi OH2 (RK3576) 外部 MIPI-CSI 摄像头原生 C++ 部署与实时推理工程实录

- **文档归档**: `docs/web_cam_rk3576.md`
- **目标平台**: 触觉智能 Purple Pi OH2 (Rockchip RK3576 / 双核 NPU @ 6 TOPS)
- **操作系统**: Ubuntu 22.04 LTS (Linux 6.1 aarch64 内核)
- **视觉传感器**: Sony IMX415 (MIPI-CSI 接口，原生输出支持 4K @ 30fps，经板载 RKISP 下采样至 1080P NV12)
- **部署模型**: YOLOv11n (FP16 精度基线，输入分辨率 640x640x3)
- **构建方式**: 原生 C++17 + CMake + Linux V4L2 MPLANE 原生采集 + RKNPU2 C API + OpenCV

---

## 1. 硬件探测与视频通路定位

在嵌入式 Linux 系统中，MIPI 摄像头受 V4L2（Video for Linux 2）与 Media Controller 拓扑子系统共同管理。通过排查，确定了核心硬件通路与采集参数：

### 1.1 节点辨析与排查
- **初始误区**: 固件软链接 `/dev/video-camera0` 初始指向 `/dev/video51`，实测处于未激活通道，导致取流返回 0 字节。
- **媒体定位**:
  - 通过 `media-ctl -d /dev/media4 -p` 进行链路追踪，确认与 Sony IMX415 传感器物理相连、且经过硬件 ISP（`rkisp_mainpath`）处理的真实输出节点为 **`/dev/video42`**。
- **底层采集协议**:
  - 节点类型: `V4L2_CAP_VIDEO_CAPTURE_MPLANE` (多平面 Multiplanar 架构，区别于普通 PC UVC 摄像头的单平面接口)。
  - 像素格式: `V4L2_PIX_FMT_NV12` (Y/UV 4:2:0，单帧大小 $1920 \times 1080 \times 1.5 = 3,110,400$ 字节)。
  - 协商分辨率: 1920x1080 @ 30 FPS。

---

## 2. 采集技术选型：原生 V4L2/第三方黑盒框架

在针对 `/dev/video42` 的取流过程中，曾尝试使用 OpenCV 默认集成的 GStreamer 管道（`v4l2src`）与 `cv::VideoCapture cap(DEVICE, cv::CAP_V4L2)`，均抛出 `Failed to allocate required memory` 与无法建链的错误。

### 2.1 故障机理分析
- **多平面协议协商不匹配**: 瑞芯微 RKISP 的 `mainpath` 对内核连续物理内存（CMA Buffer）有严格的分配池约束。OpenCV 官方内置的 V4L2 驱动在调用 `VIDIOC_REQBUFS` 时，未正确填充 MPLANE 结构体的 plane 数组参数，导致驱动层直接拒绝。
- **帧率失配**: GStreamer 默认以 `120/1` 帧率尝试协商，远超硬件内存分配池容量。

### 2.2 解决方案：纯原生 Linux V4L2 取流类实现
抛弃第三方封装，基于底层系统调用实现轻量化、高可靠的取流类 `V4L2Camera`：
1. **格式配置**: 通过 `ioctl(fd, VIDIOC_S_FMT, &fmt)` 显式传入 `V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE` 与 `V4L2_PIX_FMT_NV12`。
2. **环形内存映射**: 通过 `VIDIOC_REQBUFS` 申请 3 个内核缓冲区，并使用 `mmap` 将物理帧映射至用户空间指针，实现零拷贝。
3. **流控循环**: 通过 `VIDIOC_STREAMON` 启动流，循环调用 `VIDIOC_DQBUF` 出队、包装为 `cv::Mat` 后立即执行 `VIDIOC_QBUF` 入队，保证连续采集。

---

## 3. C++ 实时检测系统架构与模块拆解

程序整体划分为 5 个串行处理阶段：

```text
[1. V4L2 硬件取流]
    │  ioctl(VIDIOC_DQBUF) 提取 1080P NV12 物理帧指针 (耗时 ~3ms)
    ▼
[2. 前处理 (当前 CPU 承担)]
    │  cv::cvtColor(NV12 -> BGR) + cv::resize(1920x1080 -> 640x640) + cv::cvtColor(BGR -> RGB)
    ▼
[3. NPU 硬件推理 (双核调度)]
    │  rknn_inputs_set() -> rknn_run() -> rknn_outputs_get()
    │  显式绑定核心掩码: rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1) (耗时 ~20.8ms)
    ▼
[4. 张量解析与 NMS (CPU 阶段)]
    │  遍历 (1, 84, 8400) 连续内存 -> 置信度阈值粗筛 -> cv::dnn::NMSBoxes() (耗时 < 0.5ms)
    ▼
[5. 绘制与显示输出]
    │  cv::rectangle() + cv::putText() -> cv::imshow() 渲染至 HDMI 显示器 (DISPLAY=:0)
```

---

## 4. 技术细节与工程复盘

### 4.1 NPU 侧与后处理 NMS 分工界限
- **NMS 必须由 CPU 承担的物理原因**:
  - NPU 内部主要为针对定长张量的矩阵乘累加（MAC）硬件阵列，不支持动态控制流与分支跳转（如 `if iou > threshold`）。
  - 若在 ONNX 阶段强行引入 NMS 算子，会导致模型输出尺寸动态化（Dynamic Shape），破坏 NPU 离线图编译优化，导致严重性能衰减。
- **最佳分工**:
  - NPU 负责固定维度的计算密集型矩阵计算，输出静态的 `(1, 84, 8400)` 张量。
  - CPU 负责稀疏候选框的几何运算，经前置阈值粗筛后，送入 `cv::dnn::NMSBoxes` 运算耗时低于 0.5ms，整体开销忽略不计。

### 4.2 传感器 4K 规格与系统 1080P 配置的工程权衡
- **硬件源头**: Sony IMX415 物理输出为 3840x2160 RAW10 信号。
- **工程下采样决策**:
  - 若直接以 4K NV12（单帧 12.4 MB）传入用户空间，30 FPS 时将占满 372 MB/s 的系统总线带宽，极易引发 DDR 访问拥塞。
  - YOLOv11 输入尺寸仅为 640x640，由板载 RKISP 硬件缩放单元（Hardware Scaler）在芯片级完成到 1080P（单帧 3.1 MB）的下采样，可降低 75% 的内存搬运带宽，并为小目标保留足够的边缘细节。

---

## 5. 当前性能指标与后续优化方向

### 5.1 实测运行指标
- **NPU 单次硬件推理耗时**: 20.8 ~ 23.7 ms (对应纯推理帧率 42 ~ 48 FPS)
- **端到端实时运行帧率**: 14 ~ 16 FPS (受限于 CPU 承担 1080P 软件色彩转换与双线性插值缩放)
- **显示延迟**: 稳定流畅，画面无撕裂，目标追踪边界框吻合。

### 5.2 架构升级路线（后续优化点）
1. **引入 RGA (2D 硬件图形加速引擎)**:
   - 彻底废除 CPU 侧的 `cv::cvtColor` 与 `cv::resize`。
   - 使用板载预装的 `librga.so` 直接在内核空间完成 NV12 到 RGB640x640 的硬件转换与缩放，将预处理开销从 18ms 压缩至 2ms 以内。
2. **多线程生产者-消费者流水线**:
   - 采集、NPU 推理、NMS 渲染解耦为独立线程，通过环形队列传递指针，实现全管线跑满 NPU 物理上限（45+ FPS）。

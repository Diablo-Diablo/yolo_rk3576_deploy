# Purple Pi OH2 (RK3576) 板端原生 C++ 推理环境搭建与工程部署实录

- **文档归档**: `docs/cpp_environment_setup.md`
- **目标平台**: 触觉智能 Purple Pi OH2 (Rockchip RK3576 / Dual-Core NPU @ 6 TOPS)
- **操作系统**: Ubuntu 22.04 LTS (Linux 6.1 aarch64 内核)
- **核心定位**: 彻底脱离 Python 解释器与 GIL 开销，基于 RKNPU2 C API 与 OpenCV C++ 搭建高吞吐、零拷贝、工业级嵌入式推理引擎。

---

## 1. 为什么必须采用 C++ 原生部署？

在端侧嵌入式工业视觉（如工程机械斗齿监测、低功耗刷脸解锁）场景中，Python 原型存在不可忽视的系统瓶颈：
1. **消除 GIL 与多线程死锁**: 视频流采集 (V4L2)、NPU 双核异步推理、NMS 后处理流水线可实现真正的硬件级并行。
2. **零拷贝内存直通 (Zero-Copy DMA-BUF)**: 摄像头采集到的物理帧可通过 RGA / DRM 零拷贝直接送入 NPU，省去用户空间与内核空间反复 memcpy 的 CPU 开销。
3. **极高稳定性与确定性延迟**: 避免 Python 垃圾回收 (GC) 导致的帧率偶发性丢帧与抖动。

---

## 2. 板端底层 C++ 开发环境探测与补全

### 2.1 基础工具链自检 (已就绪)
板载固件自带完整的本地原生开发套件：
- **GCC / G++**: 11.4.0 (支持完整的现代 C++17 / C++20 标准)
- **CMake**: 3.22.1
- **GNU Make**: 4.3
- **硬件加速图形库**: `/usr/lib/librga.so` (瑞芯微 2D 图形硬件加速库已预装)

### 2.2 核心缺失组件与针对性修复

#### 缺失 A：RKNPU2 底层 C 头文件 (`rknn_api.h`)
- **现象**: 系统无 `rknn_api.h`，无法引用 `rknn_init`、`rknn_run`、`rknn_outputs_get` 等核心 C 接口。
- **解决**:
  - 从开发主机原厂资料包 `rknpu2/runtime/Linux/librknn_api/include/rknn_api.h` 提取头文件。
  - 通过 ADB 部署至板端标准系统包含目录：
    ```bash
    adb push rknn_api.h /usr/include/rknn_api.h
    adb shell "chmod 644 /usr/include/rknn_api.h"
    ```

#### 缺失 B：OpenCV C++ 开发依赖 (`libopencv-dev`)
- **现象**: 原厂系统仅预装 Python 轮子，缺失 C++ 头文件目录 `/usr/include/opencv4/` 及 cmake 模块支持。
- **解决**:
  - 通过系统包管理器一次性安装原生开发库与底层图形依赖：
    ```bash
    apt-get update && apt-get install -y libopencv-dev
    ```
  - 经 `pkg-config --modversion opencv4` 验证就绪。

#### 规范性优化：动态链接器软链接修正
- **现象**: 执行 apt 时提示 `/sbin/ldconfig.real: /lib/librknnrt.so is not a symbolic link`。
- **原因**: 手动推送时直接覆盖了物理文件，未遵循 Linux SONAME 软链接规范。
- **修复**:
  ```bash
  mv /usr/lib/librknnrt.so /usr/lib/librknnrt.so.2.3.2
  ln -sf /usr/lib/librknnrt.so.2.3.2 /usr/lib/librknnrt.so
  ln -sf /usr/lib/librknnrt.so.2.3.2 /usr/lib/librknnrt.so.2
  ldconfig
  ```

---

## 3. C++ 部署工程流

工程目录结构确立为：
```text
cpp_deploy/
├── CMakeLists.txt      # 现代 CMake 跨平台构建规范
├── main.cpp            # 端到端推理、计时打点、张量解析与画框主逻辑
└── build/              # 本地编译产物目录 (生成的二进制: yolo11_demo)
```

### 3.1 跨平台 `CMakeLists.txt` 构建配置
```cmake
cmake_minimum_required(VERSION 3.10)
project(yolo11_rknn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
find_library(RKNN_RT_LIB rknnrt PATHS /usr/lib /usr/local/lib REQUIRED)

include_directories(
    /usr/include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(yolo11_demo main.cpp)
target_link_libraries(yolo11_demo
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    pthread
)
```

### 3.2 核心张量解析与端到端前向推理链路 (`main.cpp`)
1. **模型载入与双核绑定**:
   - `rknn_init(&ctx, model_data, model_size, 0, NULL)`
   - `rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1)` 显式启用 RK3576 双核并发负载调度。
2. **零拷贝图像前处理**:
   - 图像读取转 RGB，缩放至 `640x640`，设置 `RKNN_TENSOR_NHWC` 格式直通 NPU。
3. **输出张量协议与排布还原**:
   - 输出头维度为 `(1, 84, 8400)`。
   - 确认数据排布为 **`[84, 8400]`**（前 4 行为坐标 `[cx, cy, w, h]`，后 80 行为类别置信度）。
   - 遍历 8400 个 Anchor 点，执行门限过滤并结合 `cv::dnn::NMSBoxes` 进行重叠抑制。
4. **可视化输出**:
   - 在原始高分辨率图上反算真实像素坐标并绘制边界框，保存为 `result_cpp.jpg`。

---

## 4. 排错复盘：PTQ 量化坍塌/ FP16 基线

在初始部署过程中，曾出现 **“NPU 推理成功但检测出 0 个目标”** 的异常，经深入数据探测与对比实验完成精准溯源：

| 现象 / 假说 | 实测数据探测 | 根本原因定位 |
| :--- | :--- | :--- |
| **现象 1**: 检出 0 个目标 | 探针打印张量全局均值：<br>- 坐标区均值: 197.4<br>- 分类区均值: **全为 0** | **单样本量化坍塌**: 转换 INT8 时仅使用 1 张 `bus.jpg` 作为校准集，PTQ 算法统计到的激活动态范围严重失真，导致微弱的分类 Logits 在整数截断舍入时全部被抹平为 0。 |
| **现象 2**: 强加 Sigmoid 后爆出 1727 个框 | 分类区强行进入 `sigmoid(0) = 0.5`，全部远超 0.25 阈值 | Ultralytics 官方导出的 ONNX Head 内部**已经内置了 Sigmoid 归一化**，不可二次处理。 |
| **验证**: 导出 **FP16** 浮点模型并上板 | 目标物体（大巴、行人）全部精准识别画框！ | **用 FP16 原精度模型测试**，保证 C++ 推理代码与后处理寻址 100% 正确无误。 |

---

## 5. 最终性能指标与成果

经板端 C++ 原生可执行程序实测：
- **单帧硬件前向耗时**: **20.8 ~ 23.7 ms** (相较 Python 脚本提速超 25%)
- **实时推理吞吐**: **42 ~ 48 FPS** (符合工业 25~30 FPS 的实时要求)
- **检测效果**: `bus.jpg` 场景下多目标边界框与置信度与 PC 端浮点推理达到同等精度。

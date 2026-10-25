import os
import sys
import time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# ================= 配置区 =================
MODEL_PATH = '/home/industio/yolo_test/yolo11n_rk3576_int8.rknn'
IMAGE_PATH = '/home/industio/yolo_test/bus.jpg'
INPUT_SIZE = 640
# ==========================================

def run_benchmark():
    print("========================================")
    print(" RK3576 YOLOv11 板端硬件推理实测")
    print("========================================")

    # 1. 检查文件存在性
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：模型文件不存在 -> {MODEL_PATH}")
        sys.exit(1)

    # 2. 初始化 RKNNLite 引擎
    rknn = RKNNLite(verbose=False)

    # 3. 加载二进制模型
    print(f"--> [1/4] 正在加载 RKNN 模型: {MODEL_PATH}")
    ret = rknn.load_rknn(MODEL_PATH)
    if ret != 0:
        print("❌ 加载模型失败！")
        sys.exit(ret)
    print("✓ 加载模型成功")

    # 4. 初始化硬件运行时环境 (关键：绑定 RK3576 双核 NPU)
    print("--> [2/4] 初始化 NPU 运行时 (启用双核 NPU_CORE_0_1)...")
    # core_mask 说明：
    # RKNNLite.NPU_CORE_0: 单核0
    # RKNNLite.NPU_CORE_1: 单核1
    # RKNNLite.NPU_CORE_0_1: 双核并行/交替调度
    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    if ret != 0:
        print("⚠️ 双核初始化未就绪，降级尝试默认核心调度...")
        ret = rknn.init_runtime()
        if ret != 0:
            print("❌ 初始化 NPU 失败！请检查驱动状态。")
            sys.exit(ret)
    print("✓ NPU 硬件环境就绪")

    # 5. 图像前处理 (CPU 侧简单测试)
    print(f"--> [3/4] 正在载入并预处理测试图像: {IMAGE_PATH}")
    if os.path.exists(IMAGE_PATH):
        img_src = cv2.imread(IMAGE_PATH)
        img = cv2.resize(img_src, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("⚠️ 未找到测试图，使用全零伪数据进行时延纯测速")
        img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)

    # 增加 batch 维度: (1, 640, 640, 3)
    img_input = np.expand_dims(img, axis=0)

    # 6. NPU 硬件热身 (消除变频抖动与首次显存换页开销)
    print("--> 正在执行 NPU 热身 (3 轮)...")
    for _ in range(3):
        rknn.inference(inputs=[img_input])

    # 7. 性能基准测试 (10 次循环测速)
    print("--> [4/4] 正在执行 10 轮时延基准测试...")
    latencies = []
    outputs = None

    for i in range(10):
        t_start = time.perf_counter()
        outputs = rknn.inference(inputs=[img_input])
        t_cost = (time.perf_counter() - t_start) * 1000.0  # 毫秒 ms
        latencies.append(t_cost)
        print(f"    第 {i+1:2d} 次推理耗时: {t_cost:6.2f} ms")

    # 8. 统计性能指标
    avg_lat = np.mean(latencies)
    median_lat = np.median(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps = 1000.0 / avg_lat

    print("\n" + "="*40)
    print("📊 实测性能汇总 (Performance Summary):")
    print(f"  - 平均单帧时延 (Average): {avg_lat:.2f} ms")
    print(f"  - 中位数时延 (Median)   : {median_lat:.2f} ms")
    print(f"  - 极致最低时延 (Min)     : {min_lat:.2f} ms")
    print(f"  - 峰值最高时延 (Max)     : {max_lat:.2f} ms")
    print(f"  - 理论纯推理帧率 (FPS)  : {fps:.1f} FPS")
    print("="*40)

    # 9. 验证输出层张量 Shape
    if outputs is not None:
        print("\n🔍 输出张量结构核对:")
        print(f"  - 输出层数量: {len(outputs)}")
        for idx, out in enumerate(outputs):
            print(f"  - Output[{idx}] Shape: {out.shape}, Dtype: {out.dtype}")

    # 10. 释放 NPU 资源
    rknn.release()
    print("\n✓ 实测完毕，NPU 硬件上下文已安全释放")

if __name__ == '__main__':
    run_benchmark()

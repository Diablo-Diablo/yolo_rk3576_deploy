import os
import cv2
import numpy as np
from rknn.api import RKNN

ONNX_MODEL = 'yolo11n.onnx'
CALIB_DATASET = './dataset.txt'
TARGET_PLATFORM = 'rk3576'

rknn = RKNN(verbose=False)

# 1. 配置
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform=TARGET_PLATFORM
)

# 2. 载入 ONNX 并构建内存图
print("--> 载入 ONNX 并构建仿真图...")
rknn.load_onnx(model=ONNX_MODEL)
rknn.build(do_quantization=False)  # 仿真测试不需要重新跑耗时的量化，用 float 即可

# 3. 初始化 x86 仿真环境
print("--> 初始化 x86 模拟器...")
rknn.init_runtime()

# 4. 构造一张测试假数据 (1, 640, 640, 3)
img = np.zeros((640, 640, 3), dtype=np.uint8)
img_input = np.expand_dims(img, axis=0)

# 5. 执行仿真推理
print("--> 执行仿真前向推理...")
outputs = rknn.inference(inputs=[img_input])

# 6. 打印输出张量维度（关键！）
print("\n" + "="*45)
print("✓ 仿真前向传播成功！模型输出头结构：")
print(f"输出层总数: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  输出 [{i}] Shape: {out.shape}, Dtype: {out.dtype}")
print("="*45)

rknn.release()

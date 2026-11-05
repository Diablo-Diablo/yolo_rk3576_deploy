import os
import cv2
import numpy as np
from rknn.api import RKNN

# 1. 初始化 RKNN
rknn = RKNN()

# 2. 加载刚生成的 RK3576 模型
RKNN_MODEL = 'yolo11n_rk3576_int8.rknn'
print(f"--> 加载模型: {RKNN_MODEL}")
ret = rknn.load_rknn(RKNN_MODEL) #如果你调用了 load_rknn()：RKNN Toolkit2 认为你是要通过 USB/ADB 连接真实的开发板（调用板端驱动）跑推理。
if ret != 0:
    print("❌ 加载 RKNN 失败！")
    exit(ret)

# 3. 初始化仿真运行时环境 (无需物理板卡)
print("--> 初始化 x86 仿真运行时...")
ret = rknn.init_runtime()
if ret != 0:
    print("❌ 初始化仿真运行时失败！")
    exit(ret)

# 4. 准备一张测试输入 (640x640, RGB)
# 可以用之前复制的 bus.jpg，或者造一个全零假图像
if os.path.exists('./dataset/bus.jpg'):
    img = cv2.imread('./dataset/bus.jpg')
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
else:
    img = np.zeros((640, 640, 3), dtype=np.uint8)

# 增加 Batch 维度: (1, 640, 640, 3)
img_input = np.expand_dims(img, axis=0)

# 5. 执行推理
print("--> 正在执行仿真推理...")
outputs = rknn.inference(inputs=[img_input])

# 6. 打印输出张量结构
print("\n" + "="*40)
print("✓ 仿真推理执行成功！输出张量信息：")
print(f"输出层总数: {len(outputs)}")
for i, out in enumerate(outputs):
    print(f"  输出 [{i}] Shape: {out.shape}, Dtype: {out.dtype}")
print("="*40)

rknn.release()

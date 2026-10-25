import os
import sys
from rknn.api import RKNN

# ================= 配置区 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# 拼接为绝对路径，彻底避免工作路径偏差
ONNX_MODEL = os.path.join(PARENT_DIR, 'models', 'onnx' , 'yolo11n.onnx')
RKNN_MODEL = os.path.join(CURRENT_DIR, 'yolo11n_rk3576_fp16.rknn')
CALIB_DATASET = os.path.join(CURRENT_DIR, 'dataset','dataset.txt')
TARGET_PLATFORM = 'rk3576'

rknn = RKNN(verbose=False)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3576'
)
print("--> 载入 ONNX...")
rknn.load_onnx(ONNX_MODEL)

print("--> 构建 FP16 模型 (do_quantization=False)...")
# 关键：do_quantization=False，不量化，全精度！
rknn.build(do_quantization=False)

print("--> 导出 FP16 RKNN 模型...")
rknn.export_rknn(RKNN_MODEL)
rknn.release()
print("✓ FP16 模型导出完成！")

import os
import sys
from rknn.api import RKNN

# ================= 配置区 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 拼接为绝对路径，彻底避免工作路径偏差
ONNX_MODEL = os.path.join(CURRENT_DIR, 'yolo11n.onnx')
RKNN_MODEL = os.path.join(CURRENT_DIR, 'yolo11n_rk3576_int8.rknn')
CALIB_DATASET = os.path.join(CURRENT_DIR, 'dataset','dataset.txt')
TARGET_PLATFORM = 'rk3576'
# ==========================================

def convert():
    # 1. 创建 RKNN 对象
    rknn = RKNN(verbose=True)

    # 2. 模型配置
    # YOLO 图像预处理公式: normalized = (pixel - mean) / std
    # mean=[0,0,0], std=[255,255,255] 等价于将输入 0~255 映射到 0.0~1.0
    print('--> 正在配置模型参数...')
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=TARGET_PLATFORM,
        quantized_algorithm='normal',      # 量化算法：normal 兼顾速度与精度
        optimization_level=3               # 最高级别图优化（算子融合、冗余节点消除）
    )
    print('--> 配置完成')

    # 3. 加载 ONNX 模型
    print(f'--> 正在载入 ONNX 模型: {ONNX_MODEL}')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('❌ 载入 ONNX 失败！请检查模型文件路径或 opset。')
        sys.exit(ret)
    print('--> 载入 ONNX 成功')

    # 4. 构建并量化模型（生成针对 RK3576 NPU 的指令流）
    print('--> 正在构建 RKNN 模型 (执行 INT8 非对称量化)...')
    ret = rknn.build(do_quantization=True, dataset=CALIB_DATASET) 
    if ret != 0:
        print('❌ 构建 RKNN 失败！')
        sys.exit(ret)
    print('--> 构建完成')

    # 5. 导出 .rknn 模型文件
    print(f'--> 正在导出 RKNN 模型: {RKNN_MODEL}')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('❌ 导出 RKNN 模型失败！')
        sys.exit(ret)
    print(f'✓ 转换成功！模型已保存为: {RKNN_MODEL}')

    # 6. 释放资源
    rknn.release()

if __name__ == '__main__':
    convert()

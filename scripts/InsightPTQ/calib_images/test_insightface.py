from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

root = Path(r"D:\face_train\datasets\wider_mafa_yolo\images\val\wider")
images = list(root.rglob("*.jpg"))
if not images:
    raise SystemExit(f"No JPG images found under {root}")

image_path = images[0]  # 先用找到的第一张 WIDERFace 图
image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Cannot read image: {image_path}")

# 仅用于验证 ONNX 能否运行：整图缩放，不做人脸检测或对齐。
rgb = cv2.cvtColor(cv2.resize(image, (112, 112)), cv2.COLOR_BGR2RGB)
x = (rgb.astype(np.float32) - 127.5) / 127.5
x = np.transpose(x, (2, 0, 1))[None, ...]

model = r"D:\PTQ\models\w600k_mbf.onnx"
session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = session.get_inputs()[0]
outputs = session.run(None, {inp.name: x})

print("image:", image_path)
print("input:", inp.name, x.shape, x.dtype)
for meta, value in zip(session.get_outputs(), outputs):
    print("output:", meta.name, value.shape, value.dtype)
    print("embedding L2 norm:", float(np.linalg.norm(value)))
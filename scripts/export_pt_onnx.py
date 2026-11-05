from pathlib import Path
from ultralytics import YOLO


WEIGHTS = Path(
    "/mnt/d/yolo_rk3576_deploy/models/yolov11n/pt/yolo11n_face.pt"
)


def main() -> None:
    """Export the face detector checkpoint to static-shape ONNX."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    output = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=12,
    )
    print(f"ONNX exported to: {output}")


if __name__ == "__main__":
    main()

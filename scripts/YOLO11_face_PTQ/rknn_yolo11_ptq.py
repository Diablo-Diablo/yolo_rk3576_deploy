from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rknn.api import RKNN

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_LIST = SCRIPT_DIR / "rknn_calib_1000" / "dataset.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO11 ONNX detector to RK3576 RKNN (FP16 or INT8)."
    )
    parser.add_argument("--onnx", required=False, type=Path, help="Exact ONNX model to convert", default=Path(__file__).parent.parent.parent / "models" / "yolov11n" / "yolo_face" / "yolo11n_face_split.onnx")
    parser.add_argument("--mode", required=False, choices=("fp16", "int8"),default="int8", help="Quantization mode: fp16 or int8")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-list", type=Path, default=DEFAULT_CALIBRATION_LIST)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--std", type=float, nargs=3, default=(255.0, 255.0, 255.0))
    parser.add_argument("--target", default="rk3576")
    return parser.parse_args()


def validate_calibration_list(list_path: Path) -> list[str]:
    if not list_path.is_file():
        raise FileNotFoundError(f"Calibration list not found: {list_path}")
    entries = [line.strip() for line in list_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"Calibration list is empty: {list_path}")
    missing = [entry for entry in entries if not Path(entry).is_file()]
    if missing:
        raise FileNotFoundError(f"Calibration image missing ({len(missing)} total); first: {missing[0]}")
    return entries


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if len(args.mean) != 3 or len(args.std) != 3 or any(value == 0 for value in args.std):
        raise ValueError("--mean and --std must each contain three values; std values cannot be zero")

    calibration_entries: list[str] = []
    if args.mode == "int8":
        calibration_entries = validate_calibration_list(args.calibration_list)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{onnx_path.stem}_rk3576_{args.mode}.rknn"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {output_path}")

    print(f"ONNX: {onnx_path}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Target: {args.target}")
    print(f"Output: {output_path}")
    if args.mode == "int8":
        print(f"Calibration images: {len(calibration_entries)} from {args.calibration_list}")

    rknn = RKNN(verbose=True)
    try:
        config_kwargs: dict[str, object] = {
            "mean_values": [list(args.mean)],
            "std_values": [list(args.std)],
            "target_platform": args.target,
        }
        if args.mode == "int8":
            config_kwargs["quantized_dtype"] = "w8a8"
            config_kwargs["quantized_algorithm"] = "normal"
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN config failed with code {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"ONNX load failed with code {ret}")

        build_kwargs: dict[str, object] = {"do_quantization": args.mode == "int8"}
        if args.mode == "int8":
            build_kwargs["dataset"] = str(args.calibration_list.resolve())
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed with code {ret}")

        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed with code {ret}")
    finally:
        rknn.release()

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Export did not produce a non-empty file: {output_path}")
    print(f"Export verified: {output_path} ({output_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

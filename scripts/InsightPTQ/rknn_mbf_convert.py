"""Convert w600k_mbf.onnx to RK3576 RKNN (FP16 only).

MBF preprocessing note
-----------------------
InsightFace MobileFaceNet uses a different normalisation from YOLO:

    input_fp32 = (pixel_uint8 - 127.5) / 128.0

Expressed as RKNN config parameters:
    mean_values = [[127.5, 127.5, 127.5]]
    std_values  = [[128.0, 128.0, 128.0]]

The model expects RGB, 112x112, float32 in [-1, 1].
Do NOT reuse YOLO's mean=[[0,0,0]] / std=[[255,255,255]] here.

FP16 rationale
--------------
MBF produces a 512-d embedding followed by L2 normalisation on the host.
Quantising the backbone to INT8 compresses the activation range and can
shift the embedding direction enough to degrade cosine similarity.
FP16 preserves full numerical fidelity at roughly half the memory/bandwidth
of FP32, and RK3576 NPU runs FP16 natively.

INT8 for MBF requires a face-crop calibration set and careful accuracy
validation; that work is out of scope for this initial conversion.

Usage (run in WSL with the rknn_yolo conda env active)
------------------------------------------------------
    python /mnt/d/yolo_rk3576_deploy/scripts/insightPTQ/rknn_mbf_convert.py

    # Override paths if needed:
    python rknn_mbf_convert.py \\
        --onnx  /mnt/d/.../w600k_mbf.onnx \\
        --output-dir /mnt/d/.../insightPTQ/output_fp16 \\
        --target rk3576
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rknn.api import RKNN

# ---------------------------------------------------------------------------
# Default paths (all relative to project root, resolved at runtime)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]          # scripts/insightPTQ -> root

DEFAULT_ONNX = _PROJECT_ROOT / "models" / "insightface" / "w600k_mbf.onnx"
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "output_fp16"

# InsightFace MBF normalisation — different from YOLO, do not change.
MBF_MEAN = [127.5, 127.5, 127.5]
MBF_STD  = [128.0, 128.0, 128.0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--onnx", type=Path, default=DEFAULT_ONNX,
        help="Path to w600k_mbf.onnx (default: models/insightface/w600k_mbf.onnx)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for the output .rknn file",
    )
    p.add_argument(
        "--target", default="rk3576",
        help="RKNN target platform (default: rk3576)",
    )
    p.add_argument(
        "--verbose", action="store_true", default=True,
        help="Enable RKNN verbose logging (default: on)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.is_file():
        print(f"ERROR: ONNX not found: {onnx_path}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{onnx_path.stem}_{args.target}_fp16.rknn"

    if output_path.exists():
        print(
            f"ERROR: output already exists, refusing to overwrite:\n  {output_path}",
            file=sys.stderr,
        )
        print("Rename or delete it, or choose a different --output-dir.",
              file=sys.stderr)
        return 1

    print("=" * 60)
    print("MBF → RKNN FP16 conversion")
    print(f"  ONNX   : {onnx_path}")
    print(f"  Target : {args.target}")
    print(f"  Output : {output_path}")
    print(f"  mean   : {MBF_MEAN}  (InsightFace standard)")
    print(f"  std    : {MBF_STD}   (InsightFace standard)")
    print(f"  input  : [1, 3, 112, 112]  (batch fixed from dynamic None)")
    print("=" * 60)

    rknn = RKNN(verbose=args.verbose)
    ret_ok = True

    try:
        # --- 1. Config ---------------------------------------------------
        # dynamic_input pins the symbolic/None batch dim before load_onnx.
        # Format: [[[N, C, H, W]]] — one list per input, one shape per list.
        ret = rknn.config(
            mean_values=[MBF_MEAN],
            std_values=[MBF_STD],
            target_platform=args.target,
            dynamic_input=[[[1, 3, 112, 112]]],
            # FP16: no quantized_dtype / quantized_algorithm needed
        )
        if ret != 0:
            print(f"ERROR: rknn.config() returned {ret}", file=sys.stderr)
            ret_ok = False
            return 1

        # --- 2. Load ONNX ------------------------------------------------
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            print(f"ERROR: rknn.load_onnx() returned {ret}", file=sys.stderr)
            ret_ok = False
            return 1

        # --- 3. Build (FP16, no quantisation) ----------------------------
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print(f"ERROR: rknn.build() returned {ret}", file=sys.stderr)
            ret_ok = False
            return 1

        # --- 4. Export ---------------------------------------------------
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            print(f"ERROR: rknn.export_rknn() returned {ret}", file=sys.stderr)
            ret_ok = False
            return 1

    finally:
        rknn.release()

    # --- 5. Verify file was written -------------------------------------
    if not output_path.is_file() or output_path.stat().st_size == 0:
        print(f"ERROR: export did not produce a non-empty file: {output_path}",
              file=sys.stderr)
        return 1

    size_mb = output_path.stat().st_size / 1024 / 1024
    print()
    print("✓ Conversion successful")
    print(f"  Output : {output_path}")
    print(f"  Size   : {size_mb:.2f} MB")
    print()
    print("Next steps:")
    print("  1. adb push the .rknn to the board (e.g. /home/industio/face_test/)")
    print("  2. Run a single-image sanity check: verify output shape == (1, 512)")
    print("  3. Compare embedding cosine similarity to ONNX reference on the same crop")
    print("  4. If cosine delta < 0.01, proceed to pipeline integration")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)

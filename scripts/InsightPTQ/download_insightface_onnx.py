#!/usr/bin/env python3
"""Download the official InsightFace buffalo_sc pack and extract its recognition ONNX.

The pretrained InsightFace weights are distributed for non-commercial research only.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

# hf-mirror.com is a mainland-accessible Hugging Face endpoint. Pin the
# repository revision so the downloaded ONNX does not silently change.
URL = "https://hf-mirror.com/WePrompt/buffalo_sc/resolve/229b65be0dc7f8f33b55478772e8d3289580c356/w600k_mbf.onnx"
EXPECTED = "w600k_mbf.onnx"
EXPECTED_SHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--url", default=URL, help="Override the official model-pack URL")
    parser.add_argument("--force", action="store_true", help="Replace an existing archive/model")
    parser.add_argument("--validate", action="store_true", help="Check ONNX graph and run a zero-input smoke test (requires onnxruntime)")
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = out / "buffalo_sc.zip"
    model = out / EXPECTED

    if model.exists() and not args.force:
        print(f"Already exists: {model}")
    else:
        if not model.exists() or args.force:
            temp_model = model.with_suffix(".onnx.part")
            request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
            print(f"Downloading InsightFace recognition ONNX: {args.url}")
            try:
                with urllib.request.urlopen(request, timeout=60) as response, temp_model.open("wb") as target:
                    shutil.copyfileobj(response, target)
                temp_model.replace(model)
            except Exception:
                temp_model.unlink(missing_ok=True)
                raise
    actual_hash = sha256(model)
    if args.url == URL and actual_hash.lower() != EXPECTED_SHA256:
        model.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for pinned mirror artifact: {actual_hash}")
    print(f"Model: {model}")
    print(f"Size: {model.stat().st_size:,} bytes")
    print(f"SHA256: {actual_hash}")
    print("Input preprocessing for the usual InsightFace ArcFace model: BGR image, 112x112, (pixel - 127.5) / 127.5 or the exact normalization defined by your existing pipeline; verify model metadata/pipeline before deployment.")
    print("License: this redistributed pretrained model is subject to its model card and upstream terms; verify usage rights before deployment.")

    if args.validate:
        try:
            import onnxruntime as ort
        except ImportError:
            print("Validation requested but onnxruntime is not installed. Install it with: python -m pip install onnxruntime", file=sys.stderr)
            return 2
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        print("Inputs:")
        for item in session.get_inputs():
            print(f"  {item.name}: shape={item.shape}, type={item.type}")
        print("Outputs:")
        for item in session.get_outputs():
            print(f"  {item.name}: shape={item.shape}, type={item.type}")
        import numpy as np
        inp = session.get_inputs()[0]
        shape = [1 if not isinstance(dim, int) else dim for dim in inp.shape]
        result = session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
        print("Smoke test outputs: " + ", ".join(str(array.shape) for array in result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

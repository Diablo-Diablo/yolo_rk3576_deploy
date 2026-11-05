#!/usr/bin/env python3
"""Smoke-test a single-class YOLO face RKNN model on the target board."""

import argparse
import sys

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("image")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb, axis=0)

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(args.model)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")

        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

        outputs = rknn.inference(inputs=[input_data])
        if not outputs:
            raise RuntimeError("Inference returned no output tensors")

        print(f"input shape: {input_data.shape}, dtype: {input_data.dtype}")
        for index, output in enumerate(outputs):
            array = np.asarray(output)
            print(
                f"output[{index}]: shape={array.shape}, "
                f"dtype={array.dtype}, min={array.min():.6g}, "
                f"max={array.max():.6g}"
            )

        output = np.asarray(outputs[0])
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.shape == (5, 8400):
            scores = output[4]
        elif output.shape == (8400, 5):
            scores = output[:, 4]
        else:
            print("Unexpected output layout; skipped face-score summary.")
            return 0

        count = int(np.count_nonzero(scores >= args.conf))
        print(f"face score max: {float(scores.max()):.6f}")
        print(f"candidates >= {args.conf}: {count}")
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

#!/usr/bin/env python3
"""Print RKNN model IO attributes without starting camera inference."""

import argparse
import sys

from rknnlite.api import RKNNLite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    rknn = RKNNLite()
    try:
        ret = rknn.load_rknn(args.model)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        print("Model load and runtime initialization succeeded.")
        print(rknn.get_sdk_version())
        return 0
    finally:
        rknn.release()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

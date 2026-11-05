from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_VAL_DIR = (
    PROJECT_ROOT
    / "face_train"
    / "datasets"
    / "wider_mafa_yolo"
    / "images"
    / "val"
    / "wider"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "rknn_calib_1000"
DEFAULT_SEED = 42
DEFAULT_COUNT = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically select full-scene WIDER val images for RKNN INT8 calibration."
    )
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--dry-run", action="store_true", help="List the planned selection without copying files."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than zero")
    if not args.val_dir.is_dir():
        raise FileNotFoundError(f"WIDER val image directory not found: {args.val_dir}")
    if args.output_dir.exists() and not args.dry_run:
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {args.output_dir}"
        )

    images = sorted(
        path for path in args.val_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.count > len(images):
        raise ValueError(f"Requested {args.count} images, but only {len(images)} were found")

    selected = sorted(random.Random(args.seed).sample(images, args.count))
    print(f"Source images: {len(images)}")
    print(f"Selected images: {len(selected)}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    if args.dry_run:
        print("Dry run: no files copied.")
        return 0

    args.output_dir.mkdir(parents=True)
    dataset_list = args.output_dir / "dataset.txt"
    entries: list[str] = []
    for index, source in enumerate(selected):
        destination = args.output_dir / f"wider_val_{index:04d}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        entries.append(destination.resolve().as_posix())

    dataset_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
    listed_paths = [Path(line) for line in dataset_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [path for path in listed_paths if not path.is_file()]
    copied = sum(
        1 for path in args.output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if copied != args.count or len(listed_paths) != args.count or missing:
        raise RuntimeError(
            f"Verification failed: copied={copied}, dataset entries={len(listed_paths)}, "
            f"missing paths={len(missing)}"
        )

    print(f"Copied and verified: {copied} images")
    print(f"RKNN dataset list: {dataset_list}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

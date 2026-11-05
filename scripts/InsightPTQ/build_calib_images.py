#!/usr/bin/env python3
"""Build WIDERFace face crops for w600k_mbf INT8 calibration.

Reads YOLO labels, crops each face with a small context margin, resizes to
112x112, and writes a deterministic subset plus a manifest. The output images
are BGR PNG/JPEG files; the quantizer must use the same preprocessing as the
runtime.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(r"D:\face_train\datasets\wider_mafa_yolo"))
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--source", default="wider", help="Image/label subdirectory under the split")
    parser.add_argument("--output", type=Path, default=Path(r"D:\PTQ\calib_images"))
    parser.add_argument("--count", type=int, default=1000, help="Number of face crops; default: 1000")
    parser.add_argument("--margin", type=float, default=0.20, help="Context around each box as a fraction of box size")
    parser.add_argument("--size", type=int, default=112, help="Output crop width/height")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--min-face-px", type=int, default=16, help="Reject boxes smaller than this on either side")
    parser.add_argument("--force", action="store_true", help="Allow writing into a non-empty output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.size <= 0 or args.margin < 0 or args.min_face_px < 1:
        raise SystemExit("count/size/min-face-px must be positive and margin must be non-negative")

    image_root = args.dataset / "images" / args.split / args.source
    label_root = args.dataset / "labels" / args.split / args.source
    if not image_root.is_dir():
        raise SystemExit(f"Image directory not found: {image_root}")
    if not label_root.is_dir():
        raise SystemExit(f"Label directory not found: {label_root}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing and not args.force:
        raise SystemExit(f"Output is not empty: {output}; use --force or choose another --output")

    candidates: list[tuple[Path, int, str]] = []
    for image_path in sorted(image_root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = label_root / image_path.relative_to(image_root).with_suffix(".txt")
        if not label_path.exists():
            continue
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
        except Exception:
            continue
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id, xc, yc, bw, bh = map(float, fields)
            except ValueError:
                continue
            if int(class_id) != 0 or bw <= 0 or bh <= 0:
                continue
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            if x2 - x1 < args.min_face_px or y2 - y1 < args.min_face_px:
                continue
            candidates.append((image_path, line_no, f"{x1},{y1},{x2},{y2}"))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: min(args.count, len(candidates))]
    if len(selected) < args.count:
        print(f"Warning: requested {args.count}, found only {len(selected)} valid faces")

    manifest = output / "manifest.txt"
    written = 0
    skipped = 0
    with manifest.open("w", encoding="utf-8", newline="\n") as manifest_file:
        for index, (image_path, line_no, box_text) in enumerate(selected):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                skipped += 1
                continue
            height, width = image.shape[:2]
            x1, y1, x2, y2 = map(int, box_text.split(","))
            box_w, box_h = x2 - x1, y2 - y1
            mx, my = int(box_w * args.margin), int(box_h * args.margin)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(width, x2 + mx)
            y2 = min(height, y2 + my)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                skipped += 1
                continue
            crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
            name = f"wider_face_{written:06d}.{args.format}"
            path = output / name
            params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.format == "jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 3]
            if not cv2.imwrite(str(path), crop, params):
                skipped += 1
                continue
            manifest_file.write(f"{path}\t{image_path}\tlabel_line={line_no}\tbox={box_text}\n")
            written += 1

    print(f"Source images: {image_root}")
    print(f"Candidate face boxes: {len(candidates)}")
    print(f"Written crops: {written}")
    print(f"Skipped: {skipped}")
    print(f"Output: {output}")
    print(f"Manifest: {manifest}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())

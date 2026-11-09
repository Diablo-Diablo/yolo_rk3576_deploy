#!/usr/bin/env python3
"""Benchmark two-output RKNN face detection on a YOLO-format dataset.

Run on the RK3576 board with RKNNLite, NumPy and OpenCV installed.
Preprocessing matches the camera program: direct resize, RGB uint8.
AP uses all-points interpolation after the configured confidence cutoff;
it is not the official FDDB ellipse evaluation protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Identify floating-point boxes/scores by shape, regardless of order."""
    if outputs is None or len(outputs) != 2:
        raise ValueError("Expected exactly two RKNN outputs")
    boxes = scores = None
    for output in outputs:
        array = np.asarray(output)
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError("Need dequantized float outputs, not raw INT8")
        if not np.isfinite(array).all():
            raise ValueError("Non-finite RKNN output")
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"Unexpected output shape: {array.shape}")
        array = array[0]
        if array.shape[0] == 4:
            boxes = array.T
        elif array.shape[1] == 4:
            boxes = array
        elif array.shape[0] == 1:
            scores = array[0]
        elif array.shape[1] == 1:
            scores = array[:, 0]
        else:
            raise ValueError(f"Unknown output layout: {array.shape}")
    if boxes is None or scores is None or len(boxes) != len(scores):
        raise ValueError("Cannot pair boxes and scores")
    return boxes, scores


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU for pixel xyxy boxes."""
    intersection = np.maximum(
        np.minimum(box[2:], boxes[:, 2:])
        - np.maximum(box[:2], boxes[:, :2]), 0
    ).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    areas = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-9)


def labels(path: Path, width: int, height: int) -> np.ndarray:
    """Load required single-class YOLO labels, allowing empty files."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing label: {path}")
    if not path.read_text().strip():
        return np.empty((0, 4), dtype=np.float32)
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.shape[1] != 5 or not np.isfinite(rows).all():
        raise ValueError(f"Invalid labels: {path}")
    if np.any(rows[:, 0] != 0) or np.any(rows[:, 3:] <= 0):
        raise ValueError(f"Invalid face class or box size: {path}")
    if np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] > 1):
        raise ValueError(f"Labels not normalized: {path}")
    center = rows[:, 1:3] * [width, height]
    half = rows[:, 3:5] * [width, height] / 2
    return np.concatenate((center - half, center + half), axis=1)


def decode(boxes: np.ndarray, scores: np.ndarray, width: int,
           height: int, size: int, conf: float,
           nms: float) -> tuple[np.ndarray, np.ndarray]:
    """Filter, undo direct resize, clip and perform single-class NMS."""
    keep = (scores >= conf) & (boxes[:, 2:] > 0).all(axis=1)
    boxes, scores = boxes[keep], scores[keep]
    corners = np.concatenate(
        (boxes[:, :2] - boxes[:, 2:] / 2,
         boxes[:, :2] + boxes[:, 2:] / 2), axis=1
    ) * [width / size, height / size, width / size, height / size]
    corners[:, [0, 2]] = np.clip(corners[:, [0, 2]], 0, width)
    corners[:, [1, 3]] = np.clip(corners[:, [1, 3]], 0, height)
    valid = (corners[:, 2:] > corners[:, :2]).all(axis=1)
    corners, scores = corners[valid], scores[valid]
    rects = corners.copy()
    rects[:, 2:] -= rects[:, :2]
    indices = cv2.dnn.NMSBoxes(
        rects.tolist(), scores.tolist(), conf, nms
    ) if len(scores) else []
    indices = np.asarray(indices, dtype=int).reshape(-1)
    return corners[indices], scores[indices]


def evaluate(records: list[tuple], threshold: float) -> dict:
    """Match by descending confidence and calculate all-points AP."""
    total = sum(len(gt) for gt, _, _ in records)
    matched = [np.zeros(len(gt), dtype=bool) for gt, _, _ in records]
    predictions = sorted(
        [(float(s), i, b) for i, (_, boxes, scores) in enumerate(records)
         for b, s in zip(boxes, scores)], key=lambda row: -row[0]
    )
    hits = []
    for _, index, box in predictions:
        overlaps = iou(box, records[index][0])
        overlaps[matched[index]] = -1
        target = int(overlaps.argmax()) if len(overlaps) else -1
        hit = target >= 0 and overlaps[target] >= threshold
        if hit:
            matched[index][target] = True
        hits.append(int(hit))
    tp = np.cumsum(hits, dtype=float)
    recall = tp / max(total, 1)
    precision = tp / np.arange(1, len(tp) + 1)
    r = np.r_[0, recall, 1]
    p = np.maximum.accumulate(np.r_[1, precision, 0][::-1])[::-1]
    ap = float(np.sum(np.diff(r) * p[1:])) if total else None
    true = int(sum(hits))
    return dict(AP=ap, TP=true, FP=len(hits) - true, FN=total - true,
                precision=true / max(len(hits), 1),
                recall=true / max(total, 1))


def main() -> None:
    """Run board inference and save timings, detections and metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(
        '/home/industio/yolo_test/yolo11n_face_split_int8.rknn'))
    parser.add_argument('--dataset', type=Path, default=Path(
        '/home/industio/yolo_test/fddb_yolo'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--output', type=Path, default=Path(
        '/home/industio/yolo_test/results/fddb_int8_split'))
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--core-mask', choices=['auto', 'core0', 'core1',
                                              'both'], default='both')
    args = parser.parse_args()
    if (args.img_size <= 0 or args.warmup < 0 or args.max_images < 0
            or not 0 <= args.conf <= 1 or not 0 <= args.nms_iou <= 1):
        parser.error('Invalid size, count or threshold')
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Refusing non-empty output: {args.output}')
    image_dir = args.dataset / 'images' / args.split
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    paths = sorted(p for p in image_dir.rglob('*')
                   if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise ValueError('No benchmark images')
    # Fail before NPU initialization if a required label is absent.
    for path in paths:
        label = (args.dataset / 'labels' / args.split
                 / path.relative_to(image_dir).with_suffix('.txt'))
        if not label.is_file():
            raise FileNotFoundError(f'Missing label: {label}')
    runtime = RKNNLite(verbose=False)
    records, timing_rows, detection_rows = [], [], []
    try:
        if runtime.load_rknn(str(args.model)) != 0:
            raise RuntimeError('RKNN model load failed')
        cores = dict(core0=RKNNLite.NPU_CORE_0,
                     core1=RKNNLite.NPU_CORE_1,
                     both=RKNNLite.NPU_CORE_0_1)
        kwargs = {} if args.core_mask == 'auto' else {
            'core_mask': cores[args.core_mask]}
        if runtime.init_runtime(**kwargs) != 0:
            raise RuntimeError('RKNN runtime initialization failed')
        for index, path in enumerate(paths):
            start = time.perf_counter()
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f'Cannot read: {path}')
            height, width = image.shape[:2]
            tensor = cv2.cvtColor(cv2.resize(
                image, (args.img_size, args.img_size)), cv2.COLOR_BGR2RGB)
            tensor = np.ascontiguousarray(tensor[None])
            prepared = time.perf_counter()
            relative = path.relative_to(image_dir)
            gt = labels(args.dataset / 'labels' / args.split
                        / relative.with_suffix('.txt'), width, height)
            if index == 0:
                for _ in range(args.warmup):
                    parse_outputs(runtime.inference(inputs=[tensor]))
            before = time.perf_counter()
            outputs = runtime.inference(inputs=[tensor])
            after = time.perf_counter()
            boxes, scores = parse_outputs(outputs)
            if index == 0:
                print('Outputs:', [(o.shape, str(o.dtype)) for o in outputs])
            raw_max = float(scores.max()) if scores.size else 0.0
            pred, confidence = decode(boxes, scores, width, height,
                                      args.img_size, args.conf, args.nms_iou)
            finished = time.perf_counter()
            records.append((gt, pred, confidence))
            prep_ms = (prepared - start) * 1000
            infer_ms = (after - before) * 1000
            post_ms = (finished - after) * 1000
            timing_rows.append(dict(
                image=relative.as_posix(), gt_boxes=len(gt),
                detections=len(pred), raw_score_max=raw_max,
                candidates=int(np.count_nonzero(scores >= args.conf)),
                preprocess_ms=prep_ms, inference_ms=infer_ms,
                postprocess_ms=post_ms,
                pipeline_ms=prep_ms + infer_ms + post_ms))
            for box, score in zip(pred, confidence):
                detection_rows.append(dict(
                    image=relative.as_posix(), confidence=float(score),
                    x1=float(box[0]), y1=float(box[1]),
                    x2=float(box[2]), y2=float(box[3])))
            if (index + 1) % 50 == 0:
                print(f'{index + 1}/{len(paths)} images')
    finally:
        runtime.release()
    metrics = {f'{v / 100:.2f}': evaluate(records, v / 100)
               for v in range(50, 96, 5)}
    aps = [v['AP'] for v in metrics.values() if v['AP'] is not None]
    summary = dict(
        model=str(args.model), model_sha256=hashlib.sha256(
            args.model.read_bytes()).hexdigest(), dataset=str(args.dataset),
        split=args.split, images=len(records), core_mask=args.core_mask,
        preprocessing='direct resize, RGB uint8, no host normalization',
        confidence_threshold=args.conf, nms_iou=args.nms_iou,
        warmup=args.warmup, metrics_iou50=metrics['0.50'],
        ground_truth_boxes=sum(len(gt) for gt, _, _ in records),
        TP=metrics['0.50']['TP'], FP=metrics['0.50']['FP'],
        FN=metrics['0.50']['FN'], precision=metrics['0.50']['precision'],
        recall=metrics['0.50']['recall'], mAP50=metrics['0.50']['AP'],
        mAP50_90=(float(np.mean([
            metrics[f'{v / 100:.2f}']['AP'] for v in range(50, 91, 5)
        ])) if aps else None),
        detections_total=sum(len(boxes) for _, boxes, _ in records),
        confidence_mean=float(np.mean([
            float(scores.mean()) if len(scores) else 0.0
            for _, _, scores in records
        ])),
        inference_ms_mean=float(np.mean([
            row['inference_ms'] for row in timing_rows
        ])),
        inference_ms_median=float(np.median([
            row['inference_ms'] for row in timing_rows
        ])),
        inference_ms_p95=float(np.percentile([
            row['inference_ms'] for row in timing_rows
        ], 95)),
        AP_by_IoU=metrics, mAP50_95=float(np.mean(aps)) if aps else None,
        metric_note='All-points AP after confidence cutoff; not official FDDB',
        timing_note='RKNN inference API includes input/output handling; '
                    'pipeline excludes labels, warmup and metric calculation',
        all_scores_zero=all(row['raw_score_max'] == 0 for row in timing_rows))
    print('AP uses float boxes and unmatched-GT matching; the FP16 script '
          'uses integer NMS boxes and a different P/R matching rule. '
          'Use identical dataset/thresholds; small differences may remain.')
    for key in ('preprocess_ms', 'inference_ms', 'postprocess_ms',
                'pipeline_ms'):
        values = [r[key] for r in timing_rows]
        summary[key] = dict(mean=float(np.mean(values)),
                            p50=float(np.percentile(values, 50)),
                            p95=float(np.percentile(values, 95)),
                            p99=float(np.percentile(values, 99)))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (
        ('per_image.csv', timing_rows, list(timing_rows[0])),
        ('detections.csv', detection_rows,
         ['image', 'confidence', 'x1', 'y1', 'x2', 'y2']),
    ):
        with (args.output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    text = json.dumps(summary, indent=2, allow_nan=False)
    (args.output / 'summary.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

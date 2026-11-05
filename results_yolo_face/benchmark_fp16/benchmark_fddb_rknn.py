#!/usr/bin/env python3
"""Evaluate a single-class YOLOv11 RKNN model on a YOLO-format split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite as RKNNRuntime
    RUNTIME_KIND = "lite"
except ImportError:
    from rknn.api import RKNN as RKNNRuntime
    RUNTIME_KIND = "toolkit"


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/industio/yolo_test/yolo11n_face_fp16.rknn",
    )
    parser.add_argument(
        "--dataset",
        default="/home/industio/yolo_test/fddb_yolo",
        help="Board dataset root containing images/ and labels/.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--class-count", type=int, default=1,
                        help="Number of class-score channels in the RKNN output.")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Optional deterministic subset; 0 means all.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="/home/industio/yolo_test/results/fddb_rknn_benchmark")
    parser.add_argument("--core-mask", choices=("auto", "core0", "core1", "both"),
                        default="both")
    return parser.parse_args()


def load_labels(path: Path, width: int, height: int) -> np.ndarray:
    """Read YOLO normalized xywh labels and return pixel xyxy boxes."""
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 4), dtype=np.float32)
    rows = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if rows.shape[1] != 5:
        raise ValueError(f"Expected 5 YOLO columns in {path}, got {rows.shape[1]}")
    if np.any(rows[:, 0] != 0):
        raise ValueError(f"Expected single face class id 0 in {path}")
    cx, cy, box_w, box_h = rows[:, 1:].T
    return np.column_stack(
        ((cx - box_w / 2) * width, (cy - box_h / 2) * height,
         (cx + box_w / 2) * width, (cy + box_h / 2) * height)
    ).astype(np.float32)


def decode_output(outputs: list[np.ndarray], image_w: int, image_h: int,
                  input_size: int, conf_threshold: float,
                  nms_threshold: float, class_count: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Decode YOLO output in xywh+class-score channel-first or channel-last form."""
    if not outputs:
        raise RuntimeError("RKNN inference returned no outputs")
    raw = np.asarray(outputs[0]).squeeze()
    if raw.ndim != 2:
        raise ValueError(f"Unexpected output rank/shape: {outputs[0].shape}")
    channel_count = 4 + class_count
    if raw.shape[0] == channel_count:
        pred = raw
    elif raw.shape[1] == channel_count:
        pred = raw.T
    else:
        raise ValueError(
            f"Expected output with {channel_count} channels "
            f"(xywh + {class_count} class scores); got {raw.shape}"
        )

    class_scores = pred[4:4 + class_count]
    class_ids = class_scores.argmax(axis=0)
    scores = class_scores[class_ids, np.arange(class_scores.shape[1])]
    keep = scores >= conf_threshold
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)

    boxes = pred[:4, keep].T.astype(np.float32)
    scores = scores[keep].astype(np.float32)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    scale_x, scale_y = image_w / input_size, image_h / input_size
    xyxy = np.column_stack((x1 * scale_x, y1 * scale_y,
                            x2 * scale_x, y2 * scale_y))
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, image_w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, image_h)
    rects = [
        [int(box[0]), int(box[1]), max(1, int(box[2] - box[0])),
         max(1, int(box[3] - box[1]))]
        for box in xyxy
    ]
    indices = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf_threshold,
                                nms_threshold)
    if len(indices) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32)
    indices = np.asarray(indices).reshape(-1)
    return xyxy[indices].astype(np.float32), scores[indices]


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one xyxy box and a set of xyxy boxes."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float32)
    left_top = np.maximum(box[:2], boxes[:, :2])
    right_bottom = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0), axis=1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0
    )
    return intersection / np.maximum(area_a + area_b - intersection, 1e-9)


def average_precision(tp: np.ndarray, scores: np.ndarray,
                      num_gt: int) -> float:
    """Compute interpolated all-points average precision."""
    if num_gt == 0:
        return math.nan
    order = np.argsort(-scores, kind="stable")
    tp_cum = np.cumsum(tp[order])
    fp_cum = np.cumsum(1 - tp[order])
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) *
                        precision[changes + 1]))


def evaluate_map(records: list[dict[str, Any]], iou_threshold: float) -> float:
    """Compute dataset AP at a given IoU threshold using greedy matching."""
    num_gt = sum(len(item["gt"]) for item in records)
    pred_rows: list[tuple[float, int, np.ndarray]] = []
    for image_index, item in enumerate(records):
        for box, score in zip(item["pred"], item["scores"]):
            pred_rows.append((float(score), image_index, box))
    pred_rows.sort(key=lambda row: -row[0])
    matched = [np.zeros(len(item["gt"]), dtype=bool) for item in records]
    tp = np.zeros(len(pred_rows), dtype=np.float32)
    scores = np.zeros(len(pred_rows), dtype=np.float32)
    for index, (score, image_index, box) in enumerate(pred_rows):
        scores[index] = score
        gt = records[image_index]["gt"]
        overlaps = box_iou(box, gt)
        if len(overlaps):
            overlaps[matched[image_index]] = -1.0
            target = int(overlaps.argmax())
            if overlaps[target] >= iou_threshold:
                tp[index] = 1
                matched[image_index][target] = True
    return average_precision(tp, scores, num_gt)


def choose_core(mask_name: str) -> int | None:
    """Map user-facing core option to runtime constants."""
    if mask_name == "auto":
        return None
    names = {
        "core0": "NPU_CORE_0",
        "core1": "NPU_CORE_1",
        "both": "NPU_CORE_0_1",
    }
    return getattr(RKNNRuntime, names[mask_name])


def main() -> None:
    """Run inference, calculate detection metrics, and write result files."""
    args = parse_args()
    model_path = Path(args.model)
    data_root = Path(args.dataset)
    image_dir = data_root / "images" / args.split
    label_dir = data_root / "labels" / args.split
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {label_dir}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    rknn = RKNNRuntime(verbose=False)
    try:
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        core = choose_core(args.core_mask)
        if RUNTIME_KIND == "lite":
            ret = (rknn.init_runtime() if core is None else
                   rknn.init_runtime(core_mask=core))
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        print(f"Runtime: {RUNTIME_KIND} ({RKNNRuntime.__module__})")

        records: list[dict[str, Any]] = []
        per_image_rows: list[dict[str, Any]] = []
        warmups = 0
        inference_ms: list[float] = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            gt = load_labels(label_dir / f"{image_path.stem}.txt", width, height)
            resized = cv2.resize(image, (args.img_size, args.img_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0)
            if warmups < args.warmup:
                rknn.inference(inputs=[input_tensor])
                warmups += 1
            start = time.perf_counter()
            outputs = rknn.inference(inputs=[input_tensor])
            elapsed_ms = (time.perf_counter() - start) * 1000
            inference_ms.append(elapsed_ms)
            pred, scores = decode_output(
                outputs, width, height, args.img_size, args.conf, args.nms_iou,
                args.class_count,
            )
            records.append({"gt": gt, "pred": pred, "scores": scores})
            per_image_rows.append({
                "image": image_path.name,
                "gt_boxes": len(gt),
                "detections": len(pred),
                "confidence_mean": float(scores.mean()) if len(scores) else 0.0,
                "inference_ms": elapsed_ms,
            })

        tp = fp = fn = 0
        confidence_rows: list[dict[str, Any]] = []
        for image_path, item in zip(image_paths, records):
            matched = np.zeros(len(item["gt"]), dtype=bool)
            order = np.argsort(-item["scores"], kind="stable")
            for pred_index in order:
                overlaps = box_iou(item["pred"][pred_index], item["gt"])
                target = int(overlaps.argmax()) if len(overlaps) else -1
                is_tp = target >= 0 and overlaps[target] >= 0.50 and not matched[target]
                if is_tp:
                    matched[target] = True
                    tp += 1
                else:
                    fp += 1
                box = item["pred"][pred_index]
                confidence_rows.append({
                    "image": image_path.name,
                    "confidence": float(item["scores"][pred_index]),
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "tp_iou50": int(is_tp),
                })
            fn += int((~matched).sum())

        total_gt = sum(len(item["gt"]) for item in records)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total_gt, 1)
        iou_thresholds = np.arange(0.50, 0.901, 0.05)
        aps = {f"AP{threshold:.2f}": evaluate_map(records, float(threshold))
               for threshold in iou_thresholds}
        map50 = aps["AP0.50"]
        map50_90 = float(np.nanmean(list(aps.values())))
        summary = {
            "model": str(model_path), "dataset": str(data_root),
            "split": args.split, "images": len(records),
            "ground_truth_boxes": total_gt,
            "confidence_threshold": args.conf,
            "nms_iou_threshold": args.nms_iou,
            "metric_precision_recall_iou": 0.50,
            "TP": tp, "FP": fp, "FN": fn,
            "precision": precision, "recall": recall,
            "mAP50": map50, "mAP50_90": map50_90,
            "AP_by_IoU": aps,
            "detections_total": sum(len(item["pred"]) for item in records),
            "confidence_mean": float(np.mean([r["confidence_mean"] for r in per_image_rows])),
            "inference_ms_mean": float(np.mean(inference_ms)),
            "inference_ms_median": float(np.median(inference_ms)),
            "inference_ms_p95": float(np.percentile(inference_ms, 95)),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for filename, rows in (("per_image.csv", per_image_rows),
                               ("detections.csv", confidence_rows)):
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nResults written to: {output_dir.resolve()}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()

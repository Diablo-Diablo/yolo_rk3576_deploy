"""Collect webcam faces with YOLO, then run MBF and reference matching.

Use Windows Python for a Windows webcam. No camera is opened on import.
Face boxes are cropped with margin; five-point alignment is not performed.
Faces are compared independently: no tracking or cross-person averaging.
"""
from __future__ import annotations

import argparse
import json
import math

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]


def normalize(vector: np.ndarray) -> np.ndarray:
    """Validate and normalize one 512-dimensional embedding."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError('Expected a finite 512-dimensional embedding')
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError('Embedding has zero norm')
    return vector / norm


def embedding(session: ort.InferenceSession,
              crop: np.ndarray) -> np.ndarray:
    """Extract an MBF embedding from a BGR face crop."""
    rgb = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) - 127.5) / 128.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    if output.shape != (1, 512):
        raise ValueError(f'Unexpected MBF output: {output.shape}')
    return normalize(output)


def detect(session: ort.InferenceSession, frame: np.ndarray,
           size: int, conf: float, nms: float) -> list[tuple]:
    """Decode original or split face ONNX using direct square resize."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})
    if len(outputs) == 1 and outputs[0].shape == (1, 5, 8400):
        boxes, scores = outputs[0][0, :4].T, outputs[0][0, 4]
    elif len(outputs) == 2:
        b = [o for o in outputs if o.shape == (1, 4, 8400)]
        s = [o for o in outputs if o.shape == (1, 1, 8400)]
        if len(b) != 1 or len(s) != 1:
            raise ValueError('Unexpected split YOLO shapes')
        boxes, scores = b[0][0].T, s[0][0, 0]
    else:
        raise ValueError(f'Unexpected YOLO outputs: {[o.shape for o in outputs]}')
    mask = (scores >= conf) & np.isfinite(boxes).all(axis=1)
    boxes, scores = boxes[mask], scores[mask]
    xyxy = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2,
                           boxes[:, :2] + boxes[:, 2:] / 2), axis=1)
    xyxy *= [w / size, h / size, w / size, h / size]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    valid = (xyxy[:, 2:] > xyxy[:, :2]).all(axis=1)
    xyxy, scores = xyxy[valid], scores[valid]
    rects = xyxy.copy()
    rects[:, 2:] -= rects[:, :2]
    keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), conf, nms)
    return [(xyxy[i], float(scores[i]))
            for i in np.asarray(keep, dtype=int).reshape(-1)] if len(keep) else []


def main() -> None:
    """Capture bounded samples, release webcam, then recognize each face."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf'],
                        default='auto')
    parser.add_argument('--sample-fps', type=float, default=5)
    parser.add_argument('--duration', type=float, default=3)
    parser.add_argument('--camera-fps', type=float, default=0,
                        help='0 keeps the camera default FPS')
    parser.add_argument('--width', type=int, default=0)
    parser.add_argument('--height', type=int, default=0)
    parser.add_argument('--max-faces', type=int, default=50)
    parser.add_argument('--min-face-px', type=int, default=40)
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--yolo', type=Path, default=(
        ROOT / 'models/yolov11n/yolo_face/yolo11n_face_split.onnx'))
    parser.add_argument('--mbf', type=Path, default=(
        ROOT / 'models/insightface/w600k_mbf.onnx'))
    parser.add_argument('--reference', type=Path, default=(
        ROOT / 'face_recog/embedding_reference/owner.npy'),
                        help='One-person .npy embedding: (512,) or (1,512)')
    parser.add_argument('--threshold', type=float, required=True,
                        help='User-calibrated cosine decision threshold')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    numeric = [args.sample_fps, args.duration, args.camera_fps,
               args.margin, args.conf, args.nms_iou, args.threshold]
    if (not all(math.isfinite(v) for v in numeric)
            or min(args.sample_fps, args.duration) <= 0
            or min(args.width, args.height, args.camera_fps) < 0
            or min(args.min_face_px, args.max_faces) <= 0 or args.margin < 0
            or not 0 <= args.conf <= 1 or not 0 <= args.nms_iou <= 1
            or not -1 <= args.threshold <= 1):
        parser.error('Invalid capture settings or thresholds')
    for path in (args.yolo, args.mbf, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty or new output directory')
    raw = np.load(args.reference, allow_pickle=False)
    if raw.shape not in ((512,), (1, 512)):
        raise ValueError('Reference must represent exactly one person')
    reference = normalize(raw)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    def session(path: Path) -> ort.InferenceSession:
        return ort.InferenceSession(str(path), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    yolo, mbf = session(args.yolo), session(args.mbf)
    backend = {'auto': cv2.CAP_ANY,
               'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}[args.backend]
    camera = cv2.VideoCapture(args.camera, backend)
    cached = []
    target = math.ceil(args.sample_fps * args.duration)
    sampled = skipped = 0
    try:
        if not camera.isOpened():
            raise RuntimeError('Cannot open webcam; check index and permissions')
        if args.width:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.camera_fps:
            camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
        print('Warming up camera for 2 seconds...')
        warmup_end = time.perf_counter() + 2
        while time.perf_counter() < warmup_end:
            if not camera.read()[0]:
                raise RuntimeError('Webcam read failed during warmup')
        actual = dict(width=camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                      height=camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                      fps=camera.get(cv2.CAP_PROP_FPS))
        print('Negotiated camera:', actual)
        print(f'Target sampled frames: {target}; may drop if detection is slow')
        start = time.perf_counter()
        next_due = start
        while sampled < target:
            now = time.perf_counter()
            if now - start >= args.duration:
                break
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('Webcam read failed')
            now = time.perf_counter()
            if now - start >= args.duration:
                break
            if now < next_due:
                continue
            elapsed = now - start
            if sampled == 0:
                args.output.mkdir(parents=True, exist_ok=True)
                preview_path = args.output / 'preview.jpg'
                if not cv2.imwrite(str(preview_path), frame):
                    raise RuntimeError(f'Cannot save preview: {preview_path}')
                print(f'Preview: {preview_path.resolve()}')
            faces = detect(yolo, frame, 640, args.conf, args.nms_iou)
            sampled += 1
            for box, confidence in faces:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) < args.min_face_px:
                    continue
                h, w = frame.shape[:2]
                left = max(0, int(x1 - bw * args.margin))
                top = max(0, int(y1 - bh * args.margin))
                right = min(w, math.ceil(x2 + bw * args.margin))
                bottom = min(h, math.ceil(y2 + bh * args.margin))
                cached.append((frame[top:bottom, left:right].copy(),
                               sampled, elapsed, box.tolist(), confidence))
                if len(cached) >= args.max_faces:
                    break
            if len(cached) >= args.max_faces:
                break
            next_due += 1 / args.sample_fps
            now = time.perf_counter()
            if next_due < now:
                missed = math.ceil((now - next_due) * args.sample_fps)
                skipped += missed
                next_due += missed / args.sample_fps
        capture_seconds = time.perf_counter() - start
    finally:
        camera.release()
    del yolo
    args.output.mkdir(parents=True, exist_ok=True)
    rows, vectors = [], []
    recognize_start = time.perf_counter()
    for index, (crop, frame_id, seconds, box, confidence) in enumerate(cached):
        before = time.perf_counter()
        vector = embedding(mbf, crop)
        score = float(np.clip(vector @ reference, -1, 1))
        elapsed_ms = (time.perf_counter() - before) * 1000
        filename = f'face_{index:04d}.png'
        if not cv2.imwrite(str(args.output / filename), crop):
            raise RuntimeError('Cannot save face crop')
        vectors.append(vector)
        rows.append(dict(crop=filename, sampled_frame=frame_id,
                         capture_seconds=seconds, box_xyxy=box,
                         detection_confidence=confidence, cosine=score,
                         match=score >= args.threshold,
                         embedding_compare_ms=elapsed_ms))
    total_ms = (time.perf_counter() - recognize_start) * 1000
    np.save(args.output / 'embeddings.npy', np.stack(vectors) if vectors
            else np.empty((0, 512), dtype=np.float32))
    report = dict(target_frames=target, sampled_frames=sampled,
                  skipped_sample_slots=skipped, cached_faces=len(cached),
                  camera=actual, capture_seconds=capture_seconds,
                  recognition_phase_ms=total_ms,
                  timing_note='Recognition phase includes crop file writes; '
                              'excludes model load and final report writes',
                  reference=str(args.reference.resolve()),
                  threshold=args.threshold, results=rows,
                  note='Per-crop matches only; no tracking, alignment or '
                       'session-level identity decision. Reference must use '
                       'the same model and preprocessing.')
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'Sampled {sampled}/{target} frames; {len(cached)} faces')
    print(f'Recognition phase: {total_ms:.1f} ms')
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()

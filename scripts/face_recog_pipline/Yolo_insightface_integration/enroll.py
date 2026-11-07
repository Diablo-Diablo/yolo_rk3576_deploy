"""Enroll one person as the gallery reference embedding.

Workflow
--------
1. Open webcam, run YOLO face detection for `--duration` seconds at
   `--sample-fps`.  Every detected crop that is at least `--min-face-px`
   pixels wide/tall is kept (up to `--max-faces`).
2. Release the camera.
3. Run MBF on every cached crop, collect the 512-d unit vectors.
4. Average them, re-normalise, and overwrite
   ``D:/yolo_rk3576_deploy/face_recog/gallery/owner.npy``.

A preview image and a per-crop embedding dump are saved alongside the
gallery file so the enrolment can be audited.

Usage example
-------------
python enroll.py
python enroll.py --duration 5 --sample-fps 3 --min-face-px 60
python enroll.py --camera 1 --backend dshow
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

# Project root: scripts/face_recog_pipline/Yolo_insightface_integration/ -> root
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GALLERY = ROOT / "face_recog" / "gallery" / "owner.npy"
DEFAULT_YOLO    = ROOT / "models/yolov11n/yolo_face/yolo11n_face_split.onnx"
DEFAULT_MBF     = ROOT / "models/insightface/w600k_mbf.onnx"


# ---------------------------------------------------------------------------
# Shared helpers (same logic as webcam_yolo_mbf.py)
# ---------------------------------------------------------------------------

def normalize(vector: np.ndarray) -> np.ndarray:
    """Validate and L2-normalise one 512-dimensional embedding."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 512-dimensional embedding")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Embedding has zero norm")
    return vector / norm


def extract_embedding(session: ort.InferenceSession,
                      crop: np.ndarray) -> np.ndarray:
    """Extract a normalised MBF embedding from a BGR face crop."""
    rgb = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) - 127.5) / 128.0
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])
    output = session.run(
        None, {session.get_inputs()[0].name: tensor}
    )[0]
    if output.shape != (1, 512):
        raise ValueError(f"Unexpected MBF output shape: {output.shape}")
    return normalize(output)


def detect_faces(session: ort.InferenceSession, frame: np.ndarray,
                 size: int, conf: float, nms: float) -> list[tuple]:
    """Run YOLO and return [(xyxy_array, score), ...] in original pixel coords."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255
    )
    outputs = session.run(None, {session.get_inputs()[0].name: tensor})

    # Accept fused (1,5,8400) or split (1,4,8400)+(1,1,8400)
    if len(outputs) == 1 and outputs[0].shape == (1, 5, 8400):
        boxes, scores = outputs[0][0, :4].T, outputs[0][0, 4]
    elif len(outputs) == 2:
        b = [o for o in outputs if o.shape == (1, 4, 8400)]
        s = [o for o in outputs if o.shape == (1, 1, 8400)]
        if len(b) != 1 or len(s) != 1:
            raise ValueError("Unexpected split YOLO output shapes")
        boxes, scores = b[0][0].T, s[0][0, 0]
    else:
        raise ValueError(
            f"Unexpected YOLO outputs: {[o.shape for o in outputs]}"
        )

    mask = (scores >= conf) & np.isfinite(boxes).all(axis=1)
    boxes, scores = boxes[mask], scores[mask]
    if boxes.size == 0:
        return []

    # cx,cy,w,h -> x1,y1,x2,y2  (normalised -> pixel)
    xyxy = np.concatenate(
        (boxes[:, :2] - boxes[:, 2:] / 2,
         boxes[:, :2] + boxes[:, 2:] / 2), axis=1
    )
    xyxy *= [w / size, h / size, w / size, h / size]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    valid = (xyxy[:, 2:] > xyxy[:, :2]).all(axis=1)
    xyxy, scores = xyxy[valid], scores[valid]
    if xyxy.size == 0:
        return []

    rects = xyxy.copy()
    rects[:, 2:] -= rects[:, :2]
    keep = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), conf, nms)
    return ([(xyxy[i], float(scores[i]))
              for i in np.asarray(keep, dtype=int).reshape(-1)]
            if len(keep) else [])


# ---------------------------------------------------------------------------
# Main enrolment logic
# ---------------------------------------------------------------------------

def main() -> None:
    """Capture face crops, average their embeddings, save gallery reference."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Camera
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf"],
                        default="auto")
    parser.add_argument("--camera-fps", type=float, default=0,
                        help="0 = keep camera default FPS")
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)

    # Capture settings
    parser.add_argument("--sample-fps", type=float, default=3,
                        help="Frames to sample per second (default 3)")
    parser.add_argument("--duration", type=float, default=5,
                        help="Capture window in seconds (default 5)")
    parser.add_argument("--max-faces", type=int, default=30,
                        help="Stop early when this many crops are cached")
    parser.add_argument("--min-face-px", type=int, default=60,
                        help="Minimum face box side in pixels")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Crop margin as fraction of box size")

    # Detection
    parser.add_argument("--conf", type=float, default=0.40,
                        help="YOLO confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.45)

    # Models
    parser.add_argument("--yolo", type=Path, default=DEFAULT_YOLO)
    parser.add_argument("--mbf",  type=Path, default=DEFAULT_MBF)

    # Output
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_GALLERY,
        help="Destination .npy file (default: face_recog/gallery/owner.npy)"
    )
    parser.add_argument(
        "--min-crops", type=int, default=5,
        help="Abort if fewer than this many valid crops were collected"
    )

    args = parser.parse_args()

    # --- Validate args -------------------------------------------------------
    for path in (args.yolo, args.mbf):
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

    if args.duration <= 0 or args.sample_fps <= 0:
        parser.error("--duration and --sample-fps must be positive")
    if args.min_face_px <= 0 or args.max_faces <= 0 or args.margin < 0:
        parser.error("Invalid face size / margin settings")
    if not 0 < args.conf <= 1 or not 0 <= args.nms_iou <= 1:
        parser.error("--conf must be in (0,1], --nms-iou in [0,1]")

    # --- Load models ---------------------------------------------------------
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2

    def session(p: Path) -> ort.InferenceSession:
        return ort.InferenceSession(str(p), sess_options=opts,
                                    providers=["CPUExecutionProvider"])

    print("Loading YOLO…")
    yolo = session(args.yolo)
    print("Loading MBF…")
    mbf = session(args.mbf)

    # --- Open camera ---------------------------------------------------------
    backend_map = {"auto": cv2.CAP_ANY,
                   "dshow": cv2.CAP_DSHOW,
                   "msmf":  cv2.CAP_MSMF}
    cam = cv2.VideoCapture(args.camera, backend_map[args.backend])
    cached: list[np.ndarray] = []   # BGR face crops
    sampled = skipped = 0
    target = math.ceil(args.sample_fps * args.duration)

    try:
        if not cam.isOpened():
            raise RuntimeError("Cannot open webcam; check index / permissions")
        if args.width:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.camera_fps:
            cam.set(cv2.CAP_PROP_FPS, args.camera_fps)

        print("Warming up camera (2 s)… please face the camera.")
        warmup_end = time.perf_counter() + 2
        while time.perf_counter() < warmup_end:
            if not cam.read()[0]:
                raise RuntimeError("Webcam read failed during warm-up")

        actual = dict(
            width=cam.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
            fps=cam.get(cv2.CAP_PROP_FPS),
        )
        print(f"Camera: {actual}")
        print(f"Collecting up to {target} frames over {args.duration} s "
              f"(stop at {args.max_faces} crops)…")

        start = time.perf_counter()
        next_due = start
        preview_saved = False

        while sampled < target and len(cached) < args.max_faces:
            now = time.perf_counter()
            if now - start >= args.duration:
                break
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("Webcam read failed during capture")
            now = time.perf_counter()
            if now - start >= args.duration:
                break
            if now < next_due:
                continue

            # Save a preview from the first sampled frame
            if not preview_saved:
                preview_path = args.output.parent / "enroll_preview.jpg"
                cv2.imwrite(str(preview_path), frame)
                print(f"Preview saved: {preview_path}")
                preview_saved = True

            faces = detect_faces(yolo, frame, 640, args.conf, args.nms_iou)
            sampled += 1
            print(f"  Frame {sampled}/{target}  "
                  f"detected={len(faces)}  cached={len(cached)}", end="\r")

            for box, _conf in faces:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) < args.min_face_px:
                    continue
                h_f, w_f = frame.shape[:2]
                left   = max(0, int(x1 - bw * args.margin))
                top    = max(0, int(y1 - bh * args.margin))
                right  = min(w_f, math.ceil(x2 + bw * args.margin))
                bottom = min(h_f, math.ceil(y2 + bh * args.margin))
                cached.append(frame[top:bottom, left:right].copy())
                if len(cached) >= args.max_faces:
                    break

            # Advance sampling clock
            next_due += 1 / args.sample_fps
            now = time.perf_counter()
            if next_due < now:
                missed = math.ceil((now - next_due) * args.sample_fps)
                skipped += missed
                next_due += missed / args.sample_fps

    finally:
        cam.release()
        print()   # newline after \r progress

    del yolo   # free memory before MBF pass

    print(f"\nCapture done: {sampled} frames sampled, "
          f"{len(cached)} valid crops collected, {skipped} slots skipped.")

    if len(cached) < args.min_crops:
        raise RuntimeError(
            f"Only {len(cached)} crops collected (need >= {args.min_crops}). "
            "Move closer to camera or increase --duration / --min-face-px."
        )

    # --- Extract embeddings --------------------------------------------------
    print("Extracting embeddings…")
    vectors: list[np.ndarray] = []
    crop_dir = args.output.parent / "enroll_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    for i, crop in enumerate(cached):
        vec = extract_embedding(mbf, crop)
        vectors.append(vec)
        cv2.imwrite(str(crop_dir / f"crop_{i:04d}.jpg"), crop)
        print(f"  Crop {i+1}/{len(cached)}  "
              f"norm={np.linalg.norm(vec):.4f}", end="\r")

    print()
    del mbf

    # --- Average + re-normalise ----------------------------------------------
    matrix = np.stack(vectors)          # (N, 512)
    mean_vec = matrix.mean(axis=0)      # (512,)
    reference = normalize(mean_vec)     # unit vector

    # --- Save gallery --------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output), reference)

    # Also save per-crop embeddings for auditing
    audit_path = args.output.parent / "enroll_embeddings.npy"
    np.save(str(audit_path), matrix)

    # Simple metadata
    meta = dict(
        crops=len(vectors),
        sampled_frames=sampled,
        skipped_slots=skipped,
        camera=actual,
        output=str(args.output.resolve()),
        audit_embeddings=str(audit_path.resolve()),
        note="Reference is the L2-normalised mean of all per-crop embeddings.",
    )
    meta_path = args.output.parent / "enroll_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n✓ Gallery reference saved → {args.output.resolve()}")
    print(f"  Averaged {len(vectors)} crops  |  "
          f"norm check: {np.linalg.norm(reference):.6f} (should be 1.0)")
    print(f"  Per-crop audit embeddings: {audit_path}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()

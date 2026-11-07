"""Evaluate MBF ONNX embeddings using identity-grouped face crops.

Dataset layout: faces/person_a/*.jpg, faces/person_b/*.jpg.
Images must already be face crops (preferably five-point aligned).
No detection or alignment is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]


def preprocess(path: Path) -> np.ndarray:
    """Resize a face crop and normalize RGB pixels for MBF."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = cv2.resize(image, (112, 112))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) - 127.5) / 128.0
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])


def embed(session: ort.InferenceSession, path: Path) -> np.ndarray:
    """Return a finite, L2-normalized 512-dimensional embedding."""
    values = session.run(None, {
        session.get_inputs()[0].name: preprocess(path)
    })
    vector = np.asarray(values[0], dtype=np.float32)
    if vector.shape != (1, 512) or not np.isfinite(vector).all():
        raise ValueError(f"Unexpected embedding: {vector.shape}")
    vector = vector[0]
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError(f"Zero embedding: {path}")
    return vector / norm


def statistics(values: list[float]) -> dict:
    """Describe observed similarities without selecting a threshold."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values), "min": float(np.min(values)),
        "max": float(np.max(values)), "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    """Extract embeddings and compare all labeled image pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=(
        ROOT / "scripts/face_recog_pipline/Insightface_test/selected_crops"))
    parser.add_argument("--model", type=Path, default=(
        ROOT / "models/insightface/w600k_mbf.onnx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unlabeled", action="store_true",
                        help="Check flat face crops without identity metrics")
    args = parser.parse_args()
    if not args.model.is_file() or not args.faces.is_dir():
        parser.error("Model file and faces directory must exist")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty or new")
    samples = []
    if args.unlabeled:
        samples = [(None, path) for path in sorted(args.faces.iterdir())
                   if path.is_file() and path.suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp"}
                   and path.stem != "contact_sheet"]
    for person in sorted(args.faces.iterdir()):
        if not args.unlabeled and person.is_dir():
            samples.extend((person.name, path) for path in
                           sorted(person.rglob("*")) if path.is_file()
                           and path.suffix.lower() in {
                               ".jpg", ".jpeg", ".png", ".bmp"})
    counts = {}
    for identity, _ in samples:
        if identity is not None:
            counts[identity] = counts.get(identity, 0) + 1
    if not samples:
        parser.error("No face images found")
    if not args.unlabeled and (
            len(counts) < 2 or max(counts.values(), default=0) < 2):
        parser.error("Need >=2 identities and >=2 images for one identity")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(args.model), sess_options=options,
        providers=["CPUExecutionProvider"])
    vectors = np.stack([embed(session, path) for _, path in samples])
    rows, positive, negative = [], [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            same = (samples[i][0] == samples[j][0]
                    if not args.unlabeled else None)
            score = float(np.clip(vectors[i] @ vectors[j], -1, 1))
            if same is not None:
                (positive if same else negative).append(score)
            rows.append({
                "image_a": str(samples[i][1]),
                "image_b": str(samples[j][1]),
                "same_identity": int(same) if same is not None else "",
                "cosine": score,
            })
    report = {
        "model": str(args.model.resolve()), "identities": counts,
        "images": len(samples), "same_identity": statistics(positive),
        "different_identity": statistics(negative),
        "mode": "unlabeled" if args.unlabeled else "labeled",
        "embedding_shape": list(vectors.shape),
        "embedding_norm_min": float(np.linalg.norm(vectors, axis=1).min()),
        "embedding_norm_max": float(np.linalg.norm(vectors, axis=1).max()),
        "preprocessing": "RGB, resize112, (pixel-127.5)/128, L2 output",
        "note": "Descriptive small-sample baseline, not accuracy or FAR. "
                "Pairs are correlated; no automatic decision threshold.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", vectors)
    (args.output / "samples.json").write_text(json.dumps([
        {"identity": identity, "image": str(path.resolve())}
        for identity, path in samples
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "pairs.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_a", "image_b", "same_identity", "cosine"])
        writer.writeheader()
        writer.writerows(rows)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (args.output / "summary.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

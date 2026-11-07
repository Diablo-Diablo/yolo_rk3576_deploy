"""Select existing 112x112 face crops; do not infer identity labels."""
from pathlib import Path
import csv
import hashlib
import shutil

import cv2
import numpy as np


def main() -> None:
    """Rank unique crops by sharpness after a basic exposure filter."""
    root = Path(__file__).resolve().parents[3]
    source = root / 'scripts/InsightPTQ/calib_images'
    output = Path(__file__).resolve().parent / 'selected_crops'
    if output.exists():
        raise FileExistsError(output)
    rows, seen = [], set()
    for path in sorted(source.iterdir()):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            continue
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (112, 112):
            continue
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        clipped = float(np.mean((gray <= 5) | (gray >= 250)))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if 40 <= mean <= 215 and clipped <= 0.25:
            rows.append(dict(source=str(path), sharpness=sharpness,
                             mean_gray=mean, clipped_fraction=clipped,
                             decoded_sha256=digest))
    rows.sort(key=lambda row: (-row['sharpness'], row['source']))
    selected = rows[:50]
    if not selected:
        raise RuntimeError('No crops passed the exposure filter')
    output.mkdir()
    for index, row in enumerate(selected):
        path = Path(row['source'])
        name = f'{index:03d}_{path.name}'
        shutil.copy2(path, output / name)
        row['selected_file'] = name
        check = cv2.imread(str(output / name))
        assert check is not None and check.shape[:2] == (112, 112)
    with (output / 'selection.csv').open('w', newline='',
                                        encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    # Contact sheet is for human inspection, not identity annotation.
    tiles = []
    for row in selected:
        image = cv2.imread(str(output / row['selected_file']))
        tile = cv2.copyMakeBorder(image, 0, 20, 0, 0,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, row['selected_file'][:3], (4, 127),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        tiles.append(tile)
    while len(tiles) % 10:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + 10])
                       for i in range(0, len(tiles), 10)])
    if not cv2.imwrite(str(output / 'contact_sheet.jpg'), sheet):
        raise RuntimeError('Cannot save contact sheet')
    print(f'Exposure-qualified unique crops: {len(rows)}')
    print(f'Selected and verified: {len(selected)}')
    print(f'Output: {output}')
    print('Already cropped inputs; no recropping or identity assignment.')


if __name__ == '__main__':
    main()

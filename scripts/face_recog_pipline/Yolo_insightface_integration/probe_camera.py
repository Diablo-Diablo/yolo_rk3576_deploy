"""Probe camera brightness over three seconds for each local camera."""
import time
import cv2

for index in (0, 1):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        print('INDEX', index, 'OPEN', cap.isOpened(), flush=True)
        start = time.perf_counter()
        samples = []
        while cap.isOpened() and time.perf_counter() - start < 3:
            ok, frame = cap.read()
            if not ok:
                break
            samples.append(float(frame.mean()))
        print('INDEX', index, 'FRAMES', len(samples), 'FIRST_LAST_MEAN',
              (samples[0], samples[-1]) if samples else None, flush=True)
    finally:
        cap.release()

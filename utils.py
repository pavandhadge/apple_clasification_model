from collections import deque

import cv2
import numpy as np

window = deque(maxlen=7)

def smoothed_prediction(new_pred):
    window.append(new_pred)
    avg = np.mean(window, axis=0)
    return np.argmax(avg)

def prob_bad_from_preds(preds):
    # Support sigmoid or 2-class softmax
    if np.isscalar(preds):
        return float(preds)
    preds = np.array(preds).ravel()
    if preds.size == 1:
        return float(preds[0])
    elif preds.size >= 2:
        # assume preds[0] = prob_bad
        return float(preds[0])
    return float(preds.mean())

def sample_video_frames(cap, target_fps=3, max_frames=150):
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(input_fps / target_fps)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    idx = 0
    collected = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            collected += 1
            if collected >= max_frames:
                break
        idx += 1
    return frames, total

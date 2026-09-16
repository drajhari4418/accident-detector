"""
data/preprocessing.py
Turns raw video files into fixed-length frame sequences for training/inference.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_frame_sequence(video_path, seq_len=config.SEQUENCE_LENGTH,
                            stride=config.FRAME_SAMPLE_STRIDE,
                            frame_size=config.FRAME_SIZE):
    """
    Samples seq_len frames evenly spaced across the ENTIRE video duration.

    This matters a lot for real footage: an accident can happen anywhere
    in a clip (start, middle, end). Earlier versions of this function read
    sequentially from frame 0 and stopped once seq_len frames were
    collected — for a 30fps video with seq_len=16 and stride=2, that's
    only the first ~1 second of footage. Any accident occurring after
    that point was silently never seen by the model or motion heuristic.
    Sampling evenly across the full clip fixes that blind spot.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        # Some containers/codecs don't report frame count reliably —
        # fall back to reading sequentially and sampling every `stride`th
        # frame, same as before, as a last resort.
        cap.release()
        return _extract_sequential_fallback(video_path, seq_len, stride, frame_size)

    # Evenly spaced frame indices across the whole video, from first to
    # last frame, guaranteeing coverage of early/mid/late events alike.
    sample_indices = np.linspace(0, total_frames - 1, num=seq_len, dtype=int)

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        return None

    while len(frames) < seq_len:
        frames.append(frames[-1])

    return np.stack(frames[:seq_len], axis=0)


def _extract_sequential_fallback(video_path, seq_len, stride, frame_size):
    """Fallback for videos where frame count can't be read in advance."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, frame_size)
            frames.append(frame)
        frame_idx += 1
    cap.release()

    if len(frames) == 0:
        return None

    # Now that we know how many candidate frames exist, sample evenly
    # from among THEM (still covers full duration, just less precisely).
    if len(frames) > seq_len:
        idxs = np.linspace(0, len(frames) - 1, num=seq_len, dtype=int)
        frames = [frames[i] for i in idxs]
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return np.stack(frames[:seq_len], axis=0)


def build_dataset_index(raw_video_dir, label_map, out_dir=config.PROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)
    index_rows = []

    for sub_folder, label in label_map.items():
        folder_path = os.path.join(raw_video_dir, sub_folder)
        if not os.path.isdir(folder_path):
            print(f"[WARN] Missing folder, skipping: {folder_path}")
            continue

        video_files = [f for f in os.listdir(folder_path)
                        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for vf in tqdm(video_files, desc=f"Processing '{sub_folder}'"):
            video_path = os.path.join(folder_path, vf)
            seq = extract_frame_sequence(video_path)
            if seq is None:
                continue
            npy_name = f"{sub_folder}_{os.path.splitext(vf)[0]}.npy"
            npy_path = os.path.join(out_dir, npy_name)
            np.save(npy_path, seq)
            index_rows.append((npy_path, label))

    index_csv = os.path.join(out_dir, "index.csv")
    with open(index_csv, "w") as f:
        f.write("npy_path,label\n")
        for path, label in index_rows:
            f.write(f"{path},{label}\n")

    print(f"[INFO] Wrote {len(index_rows)} samples to {index_csv}")
    return index_csv


def extract_sliding_windows(video_path, window_len=config.SEQUENCE_LENGTH,
                             frame_stride=config.FRAME_SAMPLE_STRIDE,
                             max_windows=20, frame_size=config.FRAME_SIZE):
    """
    Scans the ENTIRE video as a series of overlapping windows, each
    `window_len` frames long (sampled at `frame_stride`), rather than one
    single global sample. This matters because accidents are brief,
    localized events that can happen anywhere in a long clip — a single
    sparse sample across the whole video dilutes short events, while
    only looking at the start misses anything later. Scanning windows
    gives fine temporal resolution AND full-duration coverage.

    Performance note: this reads the video SEQUENTIALLY, once, front to
    back, and builds all windows from that in-memory frame list. An
    earlier version used cv2.CAP_PROP_POS_FRAMES to seek to each frame
    individually — for H.264/MP4 footage, seeking requires decoding
    forward from the last keyframe every time, and with ~12 windows x 16
    frames that meant up to ~200 expensive re-decodes per video (this is
    what caused multi-minute hangs on real footage). A single sequential
    pass avoids that entirely.

    Returns a list of dicts, each with:
        - "frames": np.ndarray (window_len, H, W, 3)
        - "start_time_sec": approximate timestamp of the window's start
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Single sequential decode pass — read every raw frame once, resizing
    # immediately to keep memory bounded even for longer clips.
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        all_frames.append(frame)
    cap.release()

    total_frames = len(all_frames)
    if total_frames == 0:
        return []

    raw_window_span = window_len * frame_stride

    if total_frames <= raw_window_span:
        # Short clip — one window, sampled evenly across what we have
        idxs = np.linspace(0, total_frames - 1, num=window_len, dtype=int)
        frames = [all_frames[i] for i in idxs]
        return [{"frames": np.stack(frames, axis=0), "start_time_sec": 0.0}]

    max_start = total_frames - raw_window_span
    step = max(1, max_start // max_windows)

    starts = list(range(0, max_start + 1, step))
    if starts[-1] != max_start:
        starts.append(max_start)

    windows = []
    for start in starts:
        idxs = [start + i * frame_stride for i in range(window_len)]
        idxs = [min(i, total_frames - 1) for i in idxs]  # safety clamp
        frames = [all_frames[i] for i in idxs]
        windows.append({
            "frames": np.stack(frames, axis=0),
            "start_time_sec": round(start / fps, 2),
        })

    return windows


if __name__ == "__main__":
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    build_dataset_index(raw_dir, label_map={"accident": 1, "non_accident": 0})

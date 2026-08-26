"""
data/preprocessing.py
Turns raw video files into fixed-length frame sequences (.npy) that the
Dataset class can load quickly during training/inference.
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
    Reads a video file and returns a uniformly-sampled sequence of frames.

    Returns:
        np.ndarray of shape (seq_len, H, W, 3), dtype=uint8
        or None if the video couldn't be read / was too short.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
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
        if len(frames) >= seq_len:
            break
    cap.release()

    if len(frames) == 0:
        return None

    # Pad by repeating the last frame if the clip was shorter than seq_len
    while len(frames) < seq_len:
        frames.append(frames[-1])

    return np.stack(frames[:seq_len], axis=0)  # (seq_len, H, W, 3)


def build_dataset_index(raw_video_dir, label_map, out_dir=config.PROCESSED_DIR):
    """
    Walks raw_video_dir, extracts frame sequences for every video, saves them
    as .npy files, and writes an index CSV mapping npy_path -> label.

    Args:
        raw_video_dir: folder containing accident/ and non_accident/ subfolders
                        (or any structure — label_map controls the mapping)
        label_map: dict {sub_folder_name: label_int}, e.g.
                    {"accident": 1, "non_accident": 0}
        out_dir: where processed .npy sequences are written

    Returns:
        path to the generated index CSV
    """
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


if __name__ == "__main__":
    # Example usage:
    #   sample_data/raw/accident/*.mp4
    #   sample_data/raw/non_accident/*.mp4
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    build_dataset_index(raw_dir, label_map={"accident": 1, "non_accident": 0})

"""
make_dummy_data.py
Generates a handful of small synthetic (random-noise) video clips so you can
verify the full pipeline — preprocessing -> training -> inference — runs
correctly on your machine before you invest time collecting a real dataset.

These clips carry NO real signal (just random pixels), so a model trained on
them will not learn anything meaningful. This is purely a plumbing test.

Usage:
    python make_dummy_data.py
"""

import os
import cv2
import numpy as np

import config

N_CLIPS_PER_CLASS = 6     # a few clips per class so train/val split has enough samples
N_FRAMES = 30
FRAME_W, FRAME_H = 320, 240
FPS = 15


def make_dummy_video(path, n_frames=N_FRAMES, w=FRAME_W, h=FRAME_H, fps=FPS):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def main():
    accident_dir = os.path.join(config.DATA_DIR, "raw", "accident")
    non_accident_dir = os.path.join(config.DATA_DIR, "raw", "non_accident")
    os.makedirs(accident_dir, exist_ok=True)
    os.makedirs(non_accident_dir, exist_ok=True)

    for i in range(N_CLIPS_PER_CLASS):
        make_dummy_video(os.path.join(accident_dir, f"dummy_{i}.mp4"))
        make_dummy_video(os.path.join(non_accident_dir, f"dummy_{i}.mp4"))

    print(f"Created {N_CLIPS_PER_CLASS} dummy clips in:")
    print(f"  {accident_dir}")
    print(f"  {non_accident_dir}")
    print("\nNext steps:")
    print("  python data/preprocessing.py")
    print("  python train.py")


if __name__ == "__main__":
    main()

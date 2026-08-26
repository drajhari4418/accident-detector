"""
predict.py
CLI for running inference on a single video without launching the dashboard.

Usage:
    python predict.py --video path/to/clip.mp4
"""

import argparse
import os
import config
from controllers.controller import AccidentDetectorController


def main():
    parser = argparse.ArgumentParser(description="Run accident detection on a video clip.")
    parser.add_argument("--video", required=True, help="Path to a video file")
    parser.add_argument("--checkpoint", default="best_model.pt",
                         help="Checkpoint filename inside checkpoints/")
    args = parser.parse_args()

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, args.checkpoint)
    checkpoint_path = checkpoint_path if os.path.exists(checkpoint_path) else None
    if checkpoint_path is None:
        print("[WARN] No checkpoint found — using untrained weights (demo only).")

    controller = AccidentDetectorController(checkpoint_path=checkpoint_path)
    result = controller.predict_video(args.video)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"Video: {result['video_path']}")
    print(f"Accident probability: {result['accident_probability']:.2%}")
    print(f"Threshold: {result['threshold']:.2%}")
    print(f"Verdict: {'🚨 ACCIDENT DETECTED' if result['is_accident'] else '✅ No accident'}")


if __name__ == "__main__":
    main()

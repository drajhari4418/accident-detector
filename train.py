"""
train.py
Top-level entry point for training the accident detector.

Usage:
    1. Put raw videos in sample_data/raw/accident/ and sample_data/raw/non_accident/
    2. python data/preprocessing.py          # builds sample_data/processed/index.csv
    3. python train.py                       # trains and saves checkpoints/best_model.pt
"""

import os
import json
import config
from controllers.controller import AccidentDetectorController


def main():
    index_csv = os.path.join(config.PROCESSED_DIR, "index.csv")
    if not os.path.exists(index_csv):
        print(
            "No processed dataset found.\n"
            f"Expected: {index_csv}\n"
            "Run `python data/preprocessing.py` first (after populating "
            "sample_data/raw/accident/ and sample_data/raw/non_accident/ with videos)."
        )
        return

    controller = AccidentDetectorController(freeze_backbone=True)
    history = controller.train(num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE)

    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. History saved to {history_path}")
    print(f"Best model saved to {os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()

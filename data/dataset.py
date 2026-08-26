"""
data/dataset.py
PyTorch Dataset + DataLoader factory for accident-detection clips.
Loads the .npy frame sequences produced by preprocessing.py.
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AccidentClipDataset(Dataset):
    """
    Each item is a (sequence_length, C, H, W) tensor of frames plus a binary
    label (0 = no accident, 1 = accident).
    """

    def __init__(self, index_csv, train=True):
        self.samples = []
        with open(index_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["npy_path"], int(row["label"])))

        self.train = train
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        frames = np.load(npy_path)  # (seq_len, H, W, 3), uint8

        # Simple augmentation: random horizontal flip applied consistently
        # across the whole clip (only during training)
        if self.train and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :].copy()

        tensor_frames = torch.stack(
            [self.transform(frame) for frame in frames], dim=0
        )  # (seq_len, 3, H, W)

        return tensor_frames, torch.tensor(label, dtype=torch.long)


def get_dataloaders(index_csv=None, batch_size=config.BATCH_SIZE,
                     val_split=0.2, num_workers=2):
    """
    Builds train/val DataLoaders from the index CSV produced during
    preprocessing. Splits are random but reproducible (seeded).
    """
    index_csv = index_csv or os.path.join(config.PROCESSED_DIR, "index.csv")
    if not os.path.exists(index_csv):
        raise FileNotFoundError(
            f"No index found at {index_csv}. Run data/preprocessing.py first."
        )

    full_dataset = AccidentClipDataset(index_csv, train=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    # val set should not use train-time augmentation
    val_ds.dataset.train = False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader

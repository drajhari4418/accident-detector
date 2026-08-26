"""
controllers/controller.py
The "Controller" in MVC: mediates between the Model (models/model.py),
the Data layer (data/dataset.py), and the View (views/app.py).

Handles training, evaluation, checkpointing, and single-clip inference
so that the View never has to touch PyTorch directly.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.model import build_model
from data.dataset import get_dataloaders
from data.preprocessing import extract_frame_sequence
import torchvision.transforms as T
from data.dataset import IMAGENET_MEAN, IMAGENET_STD


class AccidentDetectorController:
    """
    Single entry point the View (Streamlit/Flask) talks to.
    Keeps model state, handles train/eval loops, and exposes a clean
    `predict_video()` method for the demo dashboard.
    """

    def __init__(self, checkpoint_path=None, freeze_backbone=True):
        self.device = config.DEVICE
        self.model = build_model(pretrained_checkpoint=checkpoint_path,
                                   freeze_backbone=freeze_backbone,
                                   device=self.device)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE,
               save_best=True, log_fn=print):
        """
        Full training loop. `log_fn` lets the View (e.g. Streamlit) redirect
        progress messages into a live UI instead of stdout.
        """
        train_loader, val_loader = get_dataloaders()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr, weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )

        best_val_f1 = 0.0
        history = []

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            val_metrics = self._evaluate(val_loader, criterion)
            scheduler.step(val_metrics["f1"])

            elapsed = time.time() - t0
            log_fn(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.3f} | "
                f"val_f1={val_metrics['f1']:.3f} | "
                f"{elapsed:.1f}s"
            )
            history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

            if save_best and val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                self.save_checkpoint("best_model.pt")
                log_fn(f"  -> New best model saved (F1={best_val_f1:.3f})")

        self.save_checkpoint("last_model.pt")
        return history

    def _train_one_epoch(self, loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        for clips, labels in loader:
            clips, labels = clips.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)

        return running_loss / len(loader.dataset)

    @torch.no_grad()
    def _evaluate(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for clips, labels in loader:
            clips, labels = clips.to(self.device), labels.to(self.device)
            logits = self.model(clips)
            loss = criterion(logits, labels)
            running_loss += loss.item() * clips.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="binary", zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)

        return {
            "loss": running_loss / len(loader.dataset),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, filename):
        path = os.path.join(config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename):
        path = os.path.join(config.CHECKPOINT_DIR, filename)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------
    # Inference (used directly by the View / dashboard)
    # ------------------------------------------------------------------
    def predict_video(self, video_path):
        """
        Runs inference on a single video file and returns a result dict
        ready for the View to display.
        """
        frames = extract_frame_sequence(video_path)
        if frames is None:
            return {"error": f"Could not read video: {video_path}"}

        clip_tensor = torch.stack(
            [self.transform(f) for f in frames], dim=0
        ).unsqueeze(0).to(self.device)  # (1, T, C, H, W)

        prob = self.model.predict_proba(clip_tensor).item()
        is_accident = prob >= config.ACCIDENT_PROB_THRESHOLD

        return {
            "video_path": video_path,
            "accident_probability": round(prob, 4),
            "is_accident": is_accident,
            "threshold": config.ACCIDENT_PROB_THRESHOLD,
        }

    def predict_frame_window(self, frame_window):
        """
        For streaming/live use: takes a list/array of already-collected
        frames (e.g. from a live camera buffer) and returns a probability.
        Used by the View for near-real-time monitoring.
        """
        clip_tensor = torch.stack(
            [self.transform(f) for f in frame_window], dim=0
        ).unsqueeze(0).to(self.device)

        prob = self.model.predict_proba(clip_tensor).item()
        return prob

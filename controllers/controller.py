"""
controllers/controller.py
The "Controller" in MVC: mediates between the Model, the Data layer, and
the View. Handles training, evaluation, checkpointing, and inference
(including the motion-heuristic ensemble for real-world robustness).
"""

import os
import sys
import time
import base64
import urllib.request
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.model import build_model
from data.dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from data.preprocessing import extract_frame_sequence, extract_sliding_windows
from utils.motion_heuristic import compute_motion_spike_score, combine_scores
import torchvision.transforms as T


# ---------------------------------------------------------------------
# Remote checkpoint download
# ---------------------------------------------------------------------
# NOTE: replace with YOUR actual GitHub Release asset URL.
CHECKPOINT_URL = "https://github.com/drajhari4418/accident-detector/releases/download/v8.0/best_model.pt"
DEFAULT_CHECKPOINT_NAME = "best_model.pt"


def ensure_checkpoint(filename=DEFAULT_CHECKPOINT_NAME, url=CHECKPOINT_URL):
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    if os.path.exists(path):
        return path
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    try:
        print(f"[INFO] Checkpoint not found locally, downloading from {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Checkpoint downloaded to {path}")
        return path
    except Exception as e:
        print(f"[WARN] Failed to download checkpoint: {e}")
        return None


class AccidentDetectorController:
    """
    Single entry point the View talks to. Keeps model state, handles
    train/eval loops, and exposes predict_video() for the dashboard.
    """

    def __init__(self, checkpoint_path=None, freeze_backbone=True, auto_download=True):
        self.device = config.DEVICE

        if checkpoint_path is None and auto_download:
            checkpoint_path = ensure_checkpoint()

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
                f"Epoch {epoch}/{num_epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.3f} | "
                f"val_f1={val_metrics['f1']:.3f} | {elapsed:.1f}s"
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
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
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
    # Inference — now with motion-heuristic ensemble
    # ------------------------------------------------------------------
    def predict_video(self, video_path):
        """
        Runs inference on a single video file by SCANNING THE ENTIRE CLIP
        as a series of overlapping windows (see
        data/preprocessing.extract_sliding_windows), rather than looking
        at only one fixed sample. This matters because accidents are
        brief, localized events that can occur anywhere in a real clip —
        a single global sample either misses late events entirely or
        dilutes brief ones. Each window is scored by the CNN+LSTM model
        ensembled with a training-free motion-spike heuristic; the
        overall verdict is driven by the window with the highest combined
        score (the clip's most "accident-like" moment).

        Returns a result dict including the peak window's timestamp, so
        you can see WHEN in the clip the system flagged something.
        """
        windows = extract_sliding_windows(video_path)
        if not windows:
            return {"error": f"Could not read video: {video_path}"}

        best = None
        best_frames = None
        window_scores = []

        for window in windows:
            frames = window["frames"]
            clip_tensor = torch.stack(
                [self.transform(f) for f in frames], dim=0
            ).unsqueeze(0).to(self.device)

            model_prob = self.model.predict_proba(clip_tensor).item()

            motion_score = 0.0
            if config.USE_MOTION_HEURISTIC:
                motion_score = compute_motion_spike_score(frames)
                combined = combine_scores(
                    model_prob, motion_score, model_weight=config.MODEL_ENSEMBLE_WEIGHT
                )
            else:
                combined = model_prob

            entry = {
                "start_time_sec": window["start_time_sec"],
                "model_probability": model_prob,
                "motion_score": motion_score,
                "combined_score": combined,
            }
            window_scores.append(entry)

            if best is None or combined > best["combined_score"]:
                best = entry
                best_frames = frames

        is_accident = best["combined_score"] >= config.ACCIDENT_PROB_THRESHOLD

        # A small JPEG thumbnail (the middle frame of the peak-scoring
        # window) as a lightweight visual record of the detection — used
        # in place of storing the full video file, since Cloud Storage
        # for Firebase now requires a paid (Blaze) plan. A single
        # compressed thumbnail is tiny (tens of KB) and fits comfortably
        # within Firestore's free tier and per-document size limits.
        thumbnail_base64 = None
        if best_frames is not None and len(best_frames) > 0:
            mid_frame = best_frames[len(best_frames) // 2]
            # Frames are stored as RGB; cv2.imencode expects BGR.
            bgr_frame = cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if success:
                thumbnail_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "video_path": video_path,
            "num_windows_scanned": len(windows),
            "model_probability": round(best["model_probability"], 4),
            "motion_score": round(best["motion_score"], 4),
            "accident_probability": round(best["combined_score"], 4),
            "peak_time_sec": best["start_time_sec"],
            "is_accident": is_accident,
            "threshold": config.ACCIDENT_PROB_THRESHOLD,
            "status_label": "ACCIDENT DETECTED" if is_accident else "NO ACCIDENT DETECTED",
            "all_window_scores": window_scores,  # useful for debugging/plotting
            "thumbnail_base64": thumbnail_base64,
        }

    def predict_frame_window(self, frame_window):
        clip_tensor = torch.stack(
            [self.transform(f) for f in frame_window], dim=0
        ).unsqueeze(0).to(self.device)
        model_prob = self.model.predict_proba(clip_tensor).item()

        if config.USE_MOTION_HEURISTIC:
            motion_score = compute_motion_spike_score(frame_window)
            return combine_scores(model_prob, motion_score, config.MODEL_ENSEMBLE_WEIGHT)
        return model_prob

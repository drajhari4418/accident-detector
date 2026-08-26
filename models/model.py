"""
models/model.py
The "Model" in MVC: a CNN+LSTM spatiotemporal classifier.

A pretrained ResNet18 extracts per-frame features, an LSTM aggregates them
across time, and a small classification head outputs accident probability.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CNNEncoder(nn.Module):
    """Per-frame feature extractor built on a pretrained ResNet18."""

    def __init__(self, feature_dim=config.CNN_FEATURE_DIM, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Drop the final FC layer; keep everything up to the global avg pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: (B, C, H, W) -> (B, feature_dim)
        feats = self.backbone(x)
        return feats.flatten(1)


class AccidentDetectorNet(nn.Module):
    """
    Full spatiotemporal model.

    Input:  (B, T, C, H, W)  — a batch of B clips, each with T frames
    Output: (B, num_classes) — raw logits
    """

    def __init__(self,
                 feature_dim=config.CNN_FEATURE_DIM,
                 hidden_dim=config.LSTM_HIDDEN_DIM,
                 num_layers=config.LSTM_NUM_LAYERS,
                 num_classes=config.NUM_CLASSES,
                 dropout=config.DROPOUT,
                 freeze_backbone=True):
        super().__init__()

        self.encoder = CNNEncoder(feature_dim=feature_dim, freeze_backbone=freeze_backbone)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # Fold time into batch dim so the CNN processes all frames at once
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)                  # (B*T, feature_dim)
        feats = feats.view(B, T, -1)              # (B, T, feature_dim)

        lstm_out, (h_n, c_n) = self.lstm(feats)   # h_n: (num_layers, B, hidden_dim)
        final_hidden = h_n[-1]                    # last layer's final hidden state: (B, hidden_dim)

        logits = self.classifier(final_hidden)    # (B, num_classes)
        return logits

    def predict_proba(self, x):
        """Convenience method returning accident-class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1]  # probability of "accident" class


def build_model(pretrained_checkpoint=None, freeze_backbone=True, device=config.DEVICE):
    """Factory function used by the Controller layer."""
    model = AccidentDetectorNet(freeze_backbone=freeze_backbone)
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        state = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from {pretrained_checkpoint}")
    return model.to(device)


if __name__ == "__main__":
    # Quick sanity check: dummy forward pass
    model = build_model()
    dummy = torch.randn(2, config.SEQUENCE_LENGTH, 3, *config.FRAME_SIZE).to(config.DEVICE)
    out = model(dummy)
    print("Output logits shape:", out.shape)  # expect (2, 2)

"""
models/model.py
The "Model" in MVC: a CNN+LSTM spatiotemporal classifier.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=config.CNN_FEATURE_DIM, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        feats = self.backbone(x)
        return feats.flatten(1)


class AccidentDetectorNet(nn.Module):
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
            input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
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
        x = x.view(B * T, C, H, W)
        feats = self.encoder(x)
        feats = feats.view(B, T, -1)
        lstm_out, (h_n, c_n) = self.lstm(feats)
        final_hidden = h_n[-1]
        logits = self.classifier(final_hidden)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs[:, 1]


def build_model(pretrained_checkpoint=None, freeze_backbone=True, device=config.DEVICE):
    model = AccidentDetectorNet(freeze_backbone=freeze_backbone)
    if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
        state = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"[INFO] Loaded weights from {pretrained_checkpoint}")
    return model.to(device)

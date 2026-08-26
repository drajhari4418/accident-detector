"""
config.py
Central configuration for the Automated Accident Detector project.
Keeping all hyperparameters and paths in one place makes the
Model / View / Controller layers easy to reconfigure without touching logic.
"""

import os
import torch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")          # raw videos / frames
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")        # extracted frame sequences
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")     # saved model weights

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Data / preprocessing
# ---------------------------------------------------------------------
FRAME_SIZE = (224, 224)     # (H, W) fed into the CNN backbone
SEQUENCE_LENGTH = 16        # number of frames per clip shown to the LSTM
FRAME_SAMPLE_STRIDE = 2     # take every Nth frame when sampling a clip
NUM_CLASSES = 2             # 0 = no accident, 1 = accident

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
CNN_BACKBONE = "resnet18"   # feature extractor (Model layer)
CNN_FEATURE_DIM = 512       # resnet18 penultimate layer size
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 1
DROPOUT = 0.4

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Inference / alerting
# ---------------------------------------------------------------------
ACCIDENT_PROB_THRESHOLD = 0.75   # probability above this = accident flagged
ALERT_CONSECUTIVE_WINDOWS = 2    # require N consecutive positive windows to fire an alert

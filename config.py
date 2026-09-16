"""
config.py
Central configuration for the Automated Accident Detector project.
"""

import os
import torch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "sample_data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Data / preprocessing
# ---------------------------------------------------------------------
FRAME_SIZE = (224, 224)
SEQUENCE_LENGTH = 16
FRAME_SAMPLE_STRIDE = 2
NUM_CLASSES = 2

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
CNN_BACKBONE = "resnet18"
CNN_FEATURE_DIM = 512
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
ACCIDENT_PROB_THRESHOLD = 0.75
ALERT_CONSECUTIVE_WINDOWS = 2

# ---------------------------------------------------------------------
# Motion heuristic ensemble (see utils/motion_heuristic.py)
# ---------------------------------------------------------------------
# While the model is trained on limited/synthetic data, we blend in a
# training-free optical-flow motion-spike signal so real footage still
# gets a meaningful signal. Raise MODEL_ENSEMBLE_WEIGHT toward 1.0 as you
# train on more real, high-quality labeled accident footage.
USE_MOTION_HEURISTIC = True
# While the model is untrained/undertrained (i.e. before you've run
# train.py on real labeled footage), its output is close to a coin flip
# and actively dilutes the more reliable motion signal. Lean toward the
# heuristic (lower weight) until you've trained on real data, then raise
# this back up as the model earns trust with real validation accuracy.
MODEL_ENSEMBLE_WEIGHT = 0.35

# ---------------------------------------------------------------------
# Firebase Auth (see controllers/auth_controller.py)
# ---------------------------------------------------------------------
# Actual FIREBASE_API_KEY value is read from .streamlit/secrets.toml —
# never hardcode it here or commit it to git.
REQUIRE_LOGIN = True

# ---------------------------------------------------------------------
# Cloud Firestore (see controllers/db_controller.py)
# ---------------------------------------------------------------------
# Saves each prediction's video + result under the logged-in user, and
# shows a history panel. Optional — the core detection feature still
# works if this isn't configured; set to False to hide the feature
# entirely (e.g. while you haven't set up Firestore yet).
ENABLE_DB_STORAGE = True

"""
tests/test_config.py
Sanity checks on config.py — catches accidental breakage of core
constants (e.g. a typo that removes NUM_CLASSES, or a threshold value
that silently becomes invalid) without needing any heavy dependencies.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def test_sequence_length_positive():
    assert config.SEQUENCE_LENGTH > 0


def test_frame_size_is_pair():
    assert len(config.FRAME_SIZE) == 2
    assert all(dim > 0 for dim in config.FRAME_SIZE)


def test_num_classes_is_binary():
    # The model/controller assume a binary accident/no-accident setup
    # throughout (e.g. predict_proba indexes class 1 directly).
    assert config.NUM_CLASSES == 2


def test_accident_threshold_in_valid_range():
    assert 0.0 <= config.ACCIDENT_PROB_THRESHOLD <= 1.0


def test_model_ensemble_weight_in_valid_range():
    assert 0.0 <= config.MODEL_ENSEMBLE_WEIGHT <= 1.0


def test_checkpoint_dir_exists():
    # config.py creates this directory on import — verify that actually
    # happened rather than assuming.
    assert os.path.isdir(config.CHECKPOINT_DIR)

"""
utils/motion_heuristic.py
A training-free auxiliary signal for accident detection.

Rationale: the CNN+LSTM model's accuracy is entirely dependent on how much
real labeled data it has seen. Until that dataset exists, its predictions
carry little signal. Sudden, sharp motion spikes (e.g. two vehicles
colliding, a vehicle abruptly changing trajectory) are a genuine physical
signature of many accidents and can be measured directly from pixel motion
via optical flow — no training required.

This module computes a "motion spike score" for a frame sequence, which
the Controller can combine with the model's probability for a more robust
real-world signal (see controllers/controller.py: predict_video).
"""

import cv2
import numpy as np


def compute_motion_spike_score(frames):
    """
    Args:
        frames: np.ndarray of shape (T, H, W, 3), RGB, uint8

    Returns:
        float in [0, 1] — higher means a sharper, more sudden motion event
        was detected somewhere in the clip (a proxy accident signal).
    """
    if frames is None or len(frames) < 2:
        return 0.0

    gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]

    flow_magnitudes = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i], gray_frames[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(float(np.mean(magnitude)))

    if len(flow_magnitudes) < 2:
        return 0.0

    flow_magnitudes = np.array(flow_magnitudes)

    # A "spike" = a frame-to-frame jump in motion magnitude that is much
    # larger than the clip's baseline motion (e.g. steady driving -> sudden
    # deceleration/impact -> chaotic post-impact motion).
    diffs = np.abs(np.diff(flow_magnitudes))
    baseline = np.median(flow_magnitudes) + 1e-6
    normalized_spike = float(np.max(diffs) / baseline)

    # Squash into [0, 1] with a soft cap; tune the divisor against real
    # footage once you have some to calibrate against.
    score = min(normalized_spike / 8.0, 1.0)
    return score


def combine_scores(model_prob, motion_score, model_weight=0.6):
    """
    Ensembles the CNN+LSTM probability with the motion heuristic.

    model_weight controls how much we trust the trained model vs. the
    heuristic. Lower this (e.g. to 0.3-0.4) while the model is still
    trained mostly on synthetic/limited data; raise it back up as you
    train on more real, high-quality labeled footage.
    """
    motion_weight = 1.0 - model_weight
    return (model_weight * model_prob) + (motion_weight * motion_score)

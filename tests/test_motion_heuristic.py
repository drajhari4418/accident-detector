"""
tests/test_motion_heuristic.py
Verifies the training-free optical-flow motion-spike heuristic actually
distinguishes a sudden chaotic motion event from steady motion — this
is the core real-world-robustness feature added to the project, so it
matters that a future refactor can't silently break it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from utils.motion_heuristic import compute_motion_spike_score, combine_scores


def test_score_is_zero_for_none_input():
    assert compute_motion_spike_score(None) == 0.0


def test_score_is_zero_for_single_frame():
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert compute_motion_spike_score(np.array([frame])) == 0.0


def test_score_returns_float_in_valid_range():
    np.random.seed(0)
    frames = np.random.randint(0, 255, (16, 64, 64, 3), dtype=np.uint8)
    score = compute_motion_spike_score(frames)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_chaotic_motion_scores_higher_than_steady_motion():
    """
    The core behavioral guarantee: a clip with steady, consistent motion
    throughout should score lower than a clip where calm motion suddenly
    becomes chaotic — the physical signature of many real collisions.
    """
    np.random.seed(0)
    base = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

    # Steady: same gentle pan direction/speed throughout.
    calm_frames = np.stack([np.roll(base, i, axis=1) for i in range(10)])

    # Sudden change: steady pan, then abrupt chaotic noise.
    chaotic_frames = np.stack(
        [np.roll(base, i, axis=1) for i in range(6)]
        + [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    )

    calm_score = compute_motion_spike_score(calm_frames)
    chaotic_score = compute_motion_spike_score(chaotic_frames)

    assert chaotic_score > calm_score


def test_combine_scores_weighting():
    # model_weight=1.0 should reduce entirely to the model's own score.
    assert combine_scores(model_prob=0.8, motion_score=0.1, model_weight=1.0) == 0.8
    # model_weight=0.0 should reduce entirely to the motion score.
    assert combine_scores(model_prob=0.8, motion_score=0.1, model_weight=0.0) == 0.1
    # Midpoint should be the simple average.
    result = combine_scores(model_prob=1.0, motion_score=0.0, model_weight=0.5)
    assert abs(result - 0.5) < 1e-9

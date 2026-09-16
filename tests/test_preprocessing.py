"""
tests/test_preprocessing.py
Tests the video frame-extraction pipeline, including the sliding-window
scanner — this is where two real bugs were caught and fixed previously
(only sampling the first ~1 second of a clip, then a severe slowdown
from per-frame video seeking). These tests guard against regressing
either fix.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest
import config
from data.preprocessing import extract_frame_sequence, extract_sliding_windows


@pytest.fixture
def synthetic_video(tmp_path):
    """Creates a short synthetic video file for testing, cleaned up
    automatically by pytest's tmp_path fixture."""
    video_path = str(tmp_path / "test_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 15, (320, 240))
    for _ in range(60):  # 4 seconds at 15fps
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


def test_extract_frame_sequence_returns_correct_shape(synthetic_video):
    frames = extract_frame_sequence(synthetic_video)
    assert frames is not None
    assert frames.shape == (config.SEQUENCE_LENGTH, *config.FRAME_SIZE, 3)


def test_extract_frame_sequence_missing_file_returns_none():
    result = extract_frame_sequence("/nonexistent/path/video.mp4")
    assert result is None


def test_sliding_windows_returns_multiple_windows(synthetic_video):
    windows = extract_sliding_windows(synthetic_video)
    assert len(windows) > 0
    for window in windows:
        assert "frames" in window
        assert "start_time_sec" in window
        assert window["frames"].shape == (config.SEQUENCE_LENGTH, *config.FRAME_SIZE, 3)


def test_sliding_windows_cover_full_duration(synthetic_video):
    """
    Regression test for the bug where only the first ~1 second of a
    video was ever analyzed. The last window's start time should be
    meaningfully into the clip, not clustered near zero.
    """
    windows = extract_sliding_windows(synthetic_video)
    start_times = [w["start_time_sec"] for w in windows]
    # The synthetic video is 4 seconds long — the scan should reach at
    # least partway through it, not stop after the first second.
    assert max(start_times) > 1.0


def test_sliding_windows_respects_max_windows_cap(synthetic_video):
    windows = extract_sliding_windows(synthetic_video, max_windows=5)
    # Allow a small overshoot for the "always include the last window"
    # guarantee, but it should stay in the same ballpark as the cap.
    assert len(windows) <= 7


def test_sliding_windows_performance_is_reasonable(synthetic_video):
    """
    Regression test for the seek-based slowdown bug (previously took
    5+ minutes on a real ~20s clip due to per-frame cv2 seeking). This
    short synthetic clip should process in well under a few seconds.
    """
    t0 = time.time()
    extract_sliding_windows(synthetic_video)
    elapsed = time.time() - t0
    assert elapsed < 10.0

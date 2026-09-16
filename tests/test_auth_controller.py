"""
tests/test_auth_controller.py
Tests the pure validation/error-handling logic in auth_controller.py —
no real network calls to Firebase (those need real credentials and
live network access, which CI doesn't have). This guards the parts
that previously caused confusing failures: a missing API key should
fail loudly and clearly, not silently.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from controllers.auth_controller import AuthController


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("FIREBASE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIREBASE_API_KEY"):
        AuthController()


def test_present_api_key_constructs_successfully(monkeypatch):
    monkeypatch.setenv("FIREBASE_API_KEY", "fake-key-for-testing")
    auth = AuthController()
    assert auth.api_key == "fake-key-for-testing"


def test_friendly_error_maps_known_firebase_codes():
    cases = {
        "EMAIL_EXISTS": "already exists",
        "INVALID_PASSWORD": "Incorrect email or password",
        "EMAIL_NOT_FOUND": "No account found",
        "WEAK_PASSWORD": "too short",
        "INVALID_EMAIL": "valid email",
    }
    for code, expected_phrase in cases.items():
        data = {"error": {"message": code}}
        message = AuthController._friendly_error(data)
        assert expected_phrase in message


def test_friendly_error_handles_none_data():
    # Simulates a non-JSON response from a gateway/proxy error page.
    message = AuthController._friendly_error(None)
    assert "non-JSON" in message or "Could not reach" in message


def test_safe_json_returns_none_for_invalid_json():
    class FakeResponse:
        status_code = 403
        def json(self):
            raise ValueError("not json")

    result = AuthController._safe_json(FakeResponse())
    assert result is None

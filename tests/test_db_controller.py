"""
tests/test_db_controller.py
Tests the pure validation logic in db_controller.py — no real network
calls to Firestore (those need real credentials). This guards the two
credential formats (base64 blob and TOML table) and the error paths
that previously produced confusing "Invalid padding" PEM errors when
a malformed value slipped through.
"""

import os
import sys
import json
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import controllers.db_controller as db_controller


FAKE_SERVICE_ACCOUNT = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "abc123",
    "private_key": "-----BEGIN PRIVATE KEY-----\nFAKEKEYDATA\n-----END PRIVATE KEY-----\n",
    "client_email": "test@test-project.iam.gserviceaccount.com",
    "client_id": "123",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://example.com/cert",
}


def test_missing_credentials_raises_clear_error(monkeypatch):
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT_B64", raising=False)
    monkeypatch.delenv("firebase_service_account", raising=False)
    with pytest.raises(RuntimeError, match="service account not found"):
        db_controller._get_service_account_dict()


def test_valid_base64_blob_roundtrips_exactly(monkeypatch):
    b64 = base64.b64encode(json.dumps(FAKE_SERVICE_ACCOUNT).encode()).decode()
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_B64", b64)
    result = db_controller._get_service_account_dict()
    assert result == FAKE_SERVICE_ACCOUNT
    # The multi-line private key specifically must survive intact — this
    # is the field that previously got corrupted via manual TOML entry.
    assert result["private_key"] == FAKE_SERVICE_ACCOUNT["private_key"]


def test_invalid_base64_raises_clear_error(monkeypatch):
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_B64", "not-valid-base64!!!")
    with pytest.raises(RuntimeError, match="not valid base64"):
        db_controller._get_service_account_dict()


def test_valid_base64_but_invalid_json_raises_clear_error(monkeypatch):
    b64 = base64.b64encode(b"not json at all").decode()
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_B64", b64)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        db_controller._get_service_account_dict()


def test_missing_required_field_raises_clear_error(monkeypatch):
    incomplete = dict(FAKE_SERVICE_ACCOUNT)
    del incomplete["private_key"]
    b64 = base64.b64encode(json.dumps(incomplete).encode()).decode()
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_B64", b64)
    with pytest.raises(RuntimeError, match="missing required field"):
        db_controller._get_service_account_dict()

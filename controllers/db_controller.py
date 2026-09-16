"""
controllers/db_controller.py
The "Controller" for persistence: wraps Firebase Admin SDK access to
Cloud Firestore for prediction history, so the View never touches the
SDK directly.

Why Firestore only (no Firebase Storage for the actual video files):
as of February 3, 2026, Cloud Storage for Firebase requires the paid
Blaze plan, even for default buckets — there is no longer a free tier
for it. Firestore itself remains free (Spark plan). So instead of
storing the full uploaded video, each prediction record includes a
small compressed JPEG thumbnail (the peak-moment frame, base64-encoded)
as a lightweight visual record — a practical stand-in that stays
entirely within Firestore's free tier and comfortably under its
1 MiB-per-document limit (typically tens of KB).

If you later upgrade to Blaze and want full video playback in history,
re-introduce Firebase Storage (or a third-party option like Cloudinary/
Backblaze B2) and store its URL alongside these Firestore records.

Uses the Firebase Admin SDK (not the client REST API) because this
Streamlit app runs entirely server-side — the Admin SDK is the correct,
standard choice for backend code like this.

Requires a Firebase service account key configured in secrets. TWO
formats are supported (see README for setup steps for each):

  RECOMMENDED — one base64-encoded blob of the whole downloaded JSON
  file, which avoids any risk of the private key getting corrupted
  during manual copy-paste into TOML (a very common source of
  "Unable to load PEM file / Invalid padding" errors):
    FIREBASE_SERVICE_ACCOUNT_B64 = "eyJ0eXBlIjogInNlcnZpY2Vfyy..."

  ALTERNATIVE — the JSON fields spelled out as a TOML table (kept for
  backward compatibility; more error-prone to set up by hand):
    [firebase_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "...@....iam.gserviceaccount.com"
    client_id = "..."
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "..."
"""

import os
import json
import base64
import binascii
import firebase_admin
from firebase_admin import credentials, firestore


def _get_secret(name):
    """Same pattern as auth_controller._get_secret — Streamlit secrets
    first, environment variable fallback, for testability outside
    Streamlit too."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
        return os.environ.get(name)
    except ImportError:
        return os.environ.get(name)
    except FileNotFoundError:
        return os.environ.get(name)


def _get_service_account_dict():
    """
    Reads Firebase service account credentials from secrets, trying the
    recommended base64-blob format first, then falling back to the
    [firebase_service_account] TOML table format. Raises a clear error
    if neither is present/valid, rather than letting the SDK fail with
    an opaque low-level message.
    """
    b64_blob = _get_secret("FIREBASE_SERVICE_ACCOUNT_B64")
    if b64_blob:
        return _decode_service_account_b64(b64_blob)

    table = _get_secret("firebase_service_account")
    if table is not None:
        return _validate_service_account_table(table)

    raise RuntimeError(
        "Firebase service account not found. Add either "
        "FIREBASE_SERVICE_ACCOUNT_B64 (recommended) or a "
        "[firebase_service_account] table to .streamlit/secrets.toml "
        "(or the equivalent Streamlit Cloud secrets). See README for "
        "how to generate this from your Firebase project."
    )


def _decode_service_account_b64(b64_blob):
    """
    Decodes a base64-encoded service account JSON blob back into a dict.
    This is the recommended path: encoding the ENTIRE downloaded JSON
    file as one opaque string sidesteps every way manual field-by-field
    TOML transcription can silently corrupt the multi-line private key
    (extra/missing newlines, smart-quote substitution, truncated
    copy-paste, etc.) — the single most common setup failure for this
    feature, surfacing as a cryptic "Unable to load PEM file / Invalid
    padding" error from the underlying crypto library.
    """
    try:
        json_bytes = base64.b64decode(b64_blob, validate=True)
    except (binascii.Error, ValueError):
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_B64 is not valid base64. Re-generate "
            "it with: python -c \"import base64; print(base64.b64encode("
            "open('path/to/your-key.json','rb').read()).decode())\" "
            "and paste the full output as one line."
        )

    try:
        sa_dict = json.loads(json_bytes)
    except json.JSONDecodeError:
        raise RuntimeError(
            "FIREBASE_SERVICE_ACCOUNT_B64 decoded to invalid JSON. Make "
            "sure you encoded the exact, unmodified service account JSON "
            "file Firebase gave you, with nothing added or removed."
        )

    return _validate_service_account_table(sa_dict)


def _validate_service_account_table(raw):
    """Shared validation for either credential format: normalizes to a
    plain dict and checks the fields firebase_admin actually needs."""
    try:
        sa_dict = dict(raw)
    except (TypeError, ValueError):
        raise RuntimeError(
            "Firebase service account secret is not a valid table/dict. "
            "If using [firebase_service_account], make sure it's a proper "
            "TOML table, not a single string value."
        )

    required_fields = ["type", "project_id", "private_key", "client_email"]
    missing = [f for f in required_fields if f not in sa_dict or not sa_dict[f]]
    if missing:
        raise RuntimeError(
            f"Firebase service account is missing required field(s): "
            f"{', '.join(missing)}. Re-check it against the downloaded "
            f"service account JSON."
        )

    return sa_dict


class DBController:
    """
    Single entry point for Firestore prediction records. Initializes
    the Admin SDK app once per process.
    """

    _app_initialized = False

    def __init__(self):
        if not DBController._app_initialized:
            sa_dict = _get_service_account_dict()
            cred = credentials.Certificate(sa_dict)
            firebase_admin.initialize_app(cred)
            DBController._app_initialized = True

        self.db = firestore.client()

    # ------------------------------------------------------------------
    # Prediction history
    # ------------------------------------------------------------------
    def save_prediction(self, user_id, user_email, result):
        """
        Writes one prediction record to the "predictions" collection.
        `result` is the dict returned by AccidentDetectorController.
        predict_video() — including its "thumbnail_base64" field, the
        small peak-moment JPEG used in place of full video storage.
        """
        doc = {
            "user_id": user_id,
            "user_email": user_email,
            "model_probability": result.get("model_probability"),
            "motion_score": result.get("motion_score"),
            "accident_probability": result.get("accident_probability"),
            "peak_time_sec": result.get("peak_time_sec"),
            "num_windows_scanned": result.get("num_windows_scanned"),
            "is_accident": result.get("is_accident"),
            "status_label": result.get("status_label"),
            "thumbnail_base64": result.get("thumbnail_base64"),
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        self.db.collection("predictions").add(doc)

    def get_user_history(self, user_id, limit=10):
        """
        Returns the `limit` most recent predictions for this user,
        newest first, as a list of dicts (including Firestore doc id).
        """
        query = (
            self.db.collection("predictions")
            .where("user_id", "==", user_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        results = []
        for doc in query.stream():
            record = doc.to_dict()
            record["doc_id"] = doc.id
            results.append(record)
        return results

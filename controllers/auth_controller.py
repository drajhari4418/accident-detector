"""
controllers/auth_controller.py
Wraps Firebase Authentication (email/password) via the Identity Toolkit
REST API — no heavy SDK/service-account setup needed, same MVC discipline
as the rest of the project (the View never talks to Firebase directly).

Why Firebase instead of Supabase: Supabase's free tier caps you at 2
active projects per organization. Firebase's free (Spark) tier does not
impose that project-count cap, and it also gives you a real-time
database for free — useful if you later want to log prediction history
per user.

Requires one secret to be configured (see README for setup):
    FIREBASE_API_KEY   (the Web API key — public by design, but still
                         keep it in secrets rather than hardcoded/committed)

In Streamlit, this is read from st.secrets. Locally outside Streamlit,
it falls back to an environment variable of the same name.
"""

import os
import requests


FIREBASE_AUTH_BASE = "https://identitytoolkit.googleapis.com/v1/accounts"


def _get_secret(name):
    """
    Reads a config value from Streamlit secrets if available, otherwise
    from an environment variable. Keeps this module importable/testable
    outside of a Streamlit runtime too.

    Only ImportError (Streamlit not installed) and StreamlitSecretNotFoundError
    (no secrets.toml file at all) are swallowed here — those are expected,
    normal fallback cases. Any other exception (e.g. a TOML syntax error
    in an existing secrets.toml, such as a missing-quotes value) is
    allowed to propagate, so a malformed secrets file fails loudly with
    a clear parse error instead of silently masquerading as "not found".
    """
    try:
        import streamlit as st
    except ImportError:
        return os.environ.get(name)

    try:
        if name in st.secrets:
            return st.secrets[name]
        return os.environ.get(name)
    except FileNotFoundError:
        # No secrets.toml present at all — fine, fall back to env var.
        return os.environ.get(name)


class AuthController:
    """
    Thin wrapper around Firebase Auth's REST API for email/password
    sign-up, sign-in, and sign-out.

    Note: Firebase's REST auth is stateless (JSON Web Tokens), so
    "sign-out" is simply discarding the token client-side — there is no
    server round-trip required, unlike Supabase's session-based model.
    """

    def __init__(self):
        self.api_key = _get_secret("FIREBASE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Firebase credentials not found. Set FIREBASE_API_KEY in "
                ".streamlit/secrets.toml (or as an environment variable). "
                "See README for how to find this in your Firebase project."
            )

    # ------------------------------------------------------------------
    def sign_up(self, email, password):
        """
        Registers a new user. Returns (success: bool, message: str).
        """
        url = f"{FIREBASE_AUTH_BASE}:signUp?key={self.api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests.RequestException as e:
            return False, f"Network error contacting Firebase: {e}"

        data = self._safe_json(response)
        if response.status_code == 200 and data is not None:
            return True, "Account created. You can now log in."
        return False, self._friendly_error(data, response)

    def sign_in(self, email, password):
        """
        Logs in an existing user. Returns (success: bool, session_or_message).

        On success, session_or_message is a dict:
            {"id_token": ..., "refresh_token": ..., "local_id": ..., "email": ...}
        """
        url = f"{FIREBASE_AUTH_BASE}:signInWithPassword?key={self.api_key}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests.RequestException as e:
            return False, f"Network error contacting Firebase: {e}"

        data = self._safe_json(response)
        if response.status_code == 200 and data is not None:
            try:
                session = {
                    "id_token": data["idToken"],
                    "refresh_token": data["refreshToken"],
                    "local_id": data["localId"],
                    "email": data["email"],
                }
                return True, session
            except KeyError:
                return False, "Unexpected response from Firebase. Please try again."
        return False, self._friendly_error(data, response)

    def sign_out(self):
        """
        Firebase REST auth is stateless — there's no server-side session
        to invalidate from the client. The View is responsible for
        discarding the stored id_token/refresh_token from session_state.
        This method exists for interface symmetry with a session-based
        provider and to leave room for token revocation later if needed.
        """
        pass

    @staticmethod
    def _safe_json(response):
        """Returns parsed JSON, or None if the response body isn't valid JSON
        (e.g. a gateway/proxy error page instead of Firebase's actual API)."""
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _friendly_error(data, response=None):
        if data is None:
            status = getattr(response, "status_code", "unknown")
            return (
                f"Could not reach Firebase properly (HTTP {status}, non-JSON response). "
                f"Check your FIREBASE_API_KEY and network connection."
            )
        try:
            msg = data.get("error", {}).get("message", "Unknown error")
        except Exception:
            msg = "Unknown error"

        mapping = {
            "EMAIL_EXISTS": "An account with this email already exists — try logging in instead.",
            "INVALID_PASSWORD": "Incorrect email or password.",
            "EMAIL_NOT_FOUND": "No account found with this email.",
            "WEAK_PASSWORD": "Password is too short (minimum 6 characters).",
            "INVALID_EMAIL": "That doesn't look like a valid email address.",
        }
        for key, friendly in mapping.items():
            if key in msg:
                return friendly
        return f"Authentication error: {msg}"

"""
views/app.py
The "View" in MVC: Streamlit dashboard, now gated behind Firebase
email/password authentication.

Run with:
    streamlit run views/app.py

Requires .streamlit/secrets.toml with:
    FIREBASE_API_KEY = "your-web-api-key"
"""

import os
import sys
import base64
import tempfile
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from controllers.controller import AccidentDetectorController
from controllers.auth_controller import AuthController
from controllers.db_controller import DBController


# ---------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------
@st.cache_resource
def load_controller():
    return AccidentDetectorController()


@st.cache_resource
def load_auth_controller():
    return AuthController()


@st.cache_resource
def load_db_controller():
    return DBController()


# ---------------------------------------------------------------------
# Auth UI
# ---------------------------------------------------------------------
def render_login_page():
    st.title("🚨 Automated Accident Detector")
    st.caption("Please log in to continue")

    auth = load_auth_controller()

    tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log In")

        if submitted:
            success, result = auth.sign_in(email, password)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = result["email"]
                st.session_state["id_token"] = result["id_token"]
                st.session_state["refresh_token"] = result["refresh_token"]
                st.session_state["local_id"] = result["local_id"]
                st.rerun()
            else:
                st.error(result)

    with tab_signup:
        with st.form("signup_form"):
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Sign Up")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, message = auth.sign_up(new_email, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def render_logout_button():
    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.get('user_email', 'unknown')}**")
        if st.button("Log Out"):
            auth = load_auth_controller()
            auth.sign_out()
            for key in ["authenticated", "user_email", "id_token", "refresh_token", "local_id"]:
                st.session_state.pop(key, None)
            st.rerun()


# ---------------------------------------------------------------------
# Main app (post-login)
# ---------------------------------------------------------------------
def render_main_app():
    render_logout_button()

    st.title("🚨 Automated Accident Detector")
    st.caption("Deep learning-based traffic accident detection (CNN + LSTM + motion analysis)")

    controller = load_controller()

    # Database is optional — the core detection feature must keep
    # working even if Firestore secrets aren't configured yet.
    db = None
    if config.ENABLE_DB_STORAGE:
        try:
            db = load_db_controller()
        except Exception as e:
            st.info(
                "Prediction history is not active yet "
                f"({e}). Detection still works — see README to enable this."
            )

    if not os.path.exists(os.path.join(config.CHECKPOINT_DIR, "best_model.pt")):
        st.warning(
            "No trained checkpoint found yet — running with an untrained model. "
            "A motion-analysis heuristic is still active, so results carry "
            "some real signal, but train on real footage for best accuracy."
        )

    st.divider()
    st.subheader("Upload a video clip")

    uploaded_file = st.file_uploader("Dashcam or CCTV clip", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        with st.spinner("Analyzing clip..."):
            result = controller.predict_video(tmp_path)

        if "error" in result:
            os.unlink(tmp_path)
            st.error(result["error"])
            return

        st.divider()
        st.subheader("Result")

        # ---- Prominent, unambiguous verdict banner ----
        if result["is_accident"]:
            st.error(f"## 🚨 {result['status_label']}")
        else:
            st.success(f"## ✅ {result['status_label']}")

        prob = result["accident_probability"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Peak Combined Score", f"{prob * 100:.1f}%")
        col2.metric("Model Signal", f"{result['model_probability'] * 100:.1f}%")
        col3.metric("Motion Signal", f"{result['motion_score'] * 100:.1f}%")

        st.progress(min(prob, 1.0))
        st.caption(
            f"Scanned {result['num_windows_scanned']} overlapping windows across the "
            f"full clip. Most accident-like moment found at **{result['peak_time_sec']}s** "
            f"into the video. Threshold: {result['threshold'] * 100:.0f}% "
            f"— combined score blends the CNN+LSTM model with a training-free "
            f"motion-spike heuristic."
        )

        # ---- Save result (+ thumbnail) to Firestore, if configured ----
        if db is not None:
            try:
                user_id = st.session_state.get("local_id", "unknown")
                user_email = st.session_state.get("user_email", "unknown")
                db.save_prediction(user_id, user_email, result)
                st.caption("✅ Saved to your prediction history.")
            except Exception as e:
                st.caption(f"⚠️ Could not save to history: {e}")

        os.unlink(tmp_path)

    # ---- Prediction history ----
    if db is not None:
        st.divider()
        st.subheader("Your Prediction History")
        try:
            user_id = st.session_state.get("local_id", "unknown")
            history = db.get_user_history(user_id, limit=10)
            if not history:
                st.caption("No predictions saved yet — upload a clip above to get started.")
            else:
                for record in history:
                    label = record.get("status_label", "UNKNOWN")
                    icon = "🚨" if record.get("is_accident") else "✅"
                    combined = record.get("accident_probability", 0) * 100
                    with st.expander(f"{icon} {label} — {combined:.1f}% combined score"):
                        thumb_b64 = record.get("thumbnail_base64")
                        if thumb_b64:
                            st.image(
                                base64.b64decode(thumb_b64),
                                caption=f"Peak moment (~{record.get('peak_time_sec', '?')}s into clip)",
                                width=320,
                            )
                        st.write(f"Peak moment: {record.get('peak_time_sec', '?')}s")
                        st.write(f"Model signal: {record.get('model_probability', 0)*100:.1f}%")
                        st.write(f"Motion signal: {record.get('motion_score', 0)*100:.1f}%")
        except Exception as e:
            st.caption(f"Could not load history: {e}")

    st.divider()
    with st.expander("About this system"):
        st.markdown(
            """
            **Architecture:** ResNet18 (per-frame CNN features) → LSTM (temporal
            aggregation) → fully-connected classifier, ensembled with an
            **optical-flow motion-spike heuristic** for real-world robustness
            even before extensive real-data training.

            **Pipeline:** video → sampled frame sequence → [CNN → LSTM] model
            probability + motion-heuristic score → weighted ensemble →
            threshold-based ACCIDENT / NO ACCIDENT verdict.

            For best accuracy, train the model on real labeled footage —
            see the project README for recommended datasets (CADP, DAD, CCD).
            """
        )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Automated Accident Detector", page_icon="🚨", layout="centered")

    if not config.REQUIRE_LOGIN:
        render_main_app()
        return

    if st.session_state.get("authenticated"):
        render_main_app()
    else:
        render_login_page()


if __name__ == "__main__":
    main()

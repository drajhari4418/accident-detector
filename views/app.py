"""
views/app.py
The "View" in MVC: a Streamlit dashboard for demoing the accident detector.
Talks ONLY to the Controller — never touches the model or dataset directly.

Run with:
    streamlit run views/app.py
"""

import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from controllers.controller import AccidentDetectorController


@st.cache_resource
def load_controller():
    return AccidentDetectorController()


def main():
    st.set_page_config(page_title="Automated Accident Detector", page_icon="🚨", layout="centered")
    st.title("🚨 Automated Accident Detector")
    st.caption("Deep learning-based traffic accident detection (CNN + LSTM)")

    controller = load_controller()

    if not os.path.exists(os.path.join(config.CHECKPOINT_DIR, "best_model.pt")):
        st.warning(
            "No trained checkpoint found yet — running with an untrained model. "
            "Predictions won't be meaningful until you run `python train.py`."
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

        os.unlink(tmp_path)

        if "error" in result:
            st.error(result["error"])
            return

        st.divider()
        st.subheader("Result")

        prob = result["accident_probability"]
        col1, col2 = st.columns(2)
        col1.metric("Accident Probability", f"{prob * 100:.1f}%")
        col2.metric("Threshold", f"{result['threshold'] * 100:.0f}%")

        st.progress(min(prob, 1.0))

        if result["is_accident"]:
            st.error("⚠️ ACCIDENT DETECTED", icon="🚨")
        else:
            st.success("✅ No accident detected", icon="✅")

    st.divider()
    with st.expander("About this system"):
        st.markdown(
            """
            **Architecture:** ResNet18 (per-frame CNN features) → LSTM (temporal
            aggregation) → fully-connected classifier.

            **Pipeline:** video → sampled frame sequence → CNN encoder →
            LSTM → accident probability → threshold-based alert.

            This is a baseline model intended as a starting point — accuracy
            depends heavily on training data quality and volume. See the
            project README for dataset recommendations (CADP, DAD, CCD).
            """
        )


if __name__ == "__main__":
    main()

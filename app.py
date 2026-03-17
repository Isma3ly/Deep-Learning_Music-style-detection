"""
Streamlit app for music genre prediction (GTZAN pipeline).
Run from project root: streamlit run app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Ensure project root is on path when running as streamlit run app.py
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from music_genre_classifier.config import DEFAULT_FILLER_TRACK
from music_genre_classifier.inference import predict_genre_from_file

MODELS_DIR = _ROOT / "models"
WEIGHTS_PATH = str(MODELS_DIR / "DNN3.weights.h5")
PCA_PATH = str(MODELS_DIR / "pca")


def main():
    st.set_page_config(page_title="Music genre classifier", page_icon="🎵", layout="centered")
    st.title("Music genre classifier")
    st.caption("GTZAN-based classifier. Place GTZAN in data/genres_original/ and run notebooks 01 and 02 first.")

    if not (MODELS_DIR / "DNN3.weights.h5").exists() or not (MODELS_DIR / "pca").exists():
        st.error("Model files not found. Run notebook 02_train_model.ipynb to produce models/DNN3.weights.h5 and models/pca.")
        return

    input_mode = st.radio("Input", ["Upload WAV file", "Path to WAV file"], horizontal=True)
    wav_path = None

    if input_mode == "Upload WAV file":
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(uploaded.read())
                wav_path = f.name
    else:
        path_input = st.text_input("Path to WAV file (relative to project root or absolute)")
        if path_input:
            p = Path(path_input)
            if not p.is_absolute():
                p = _ROOT / p
            if p.exists():
                wav_path = str(p)
            else:
                st.warning(f"File not found: {p}")

    if wav_path and st.button("Predict genre"):
        with st.spinner("Running inference…"):
            try:
                genre, prob, conf_text = predict_genre_from_file(
                    wav_path,
                    model_weights=WEIGHTS_PATH,
                    pca_path=PCA_PATH,
                    filler_track_path=DEFAULT_FILLER_TRACK,
                )
                st.success(f"{conf_text}{genre}")
                st.metric("Probability", f"{prob:.2%}")
            except Exception as e:
                st.exception(e)
            finally:
                if input_mode == "Upload WAV file" and wav_path:
                    try:
                        Path(wav_path).unlink(missing_ok=True)
                    except OSError:
                        pass


if __name__ == "__main__":
    main()

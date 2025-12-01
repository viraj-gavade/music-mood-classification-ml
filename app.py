import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.inference import MoodInference

st.set_page_config(page_title="Music Mood Classifier", layout="wide")

st.title("🎵 Music Mood Classification")
st.write("Upload an audio file and the model will predict its mood.")

model = MoodInference(model_path="model.pth")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "ogg", "flac"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/mp3')

    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ======================================================
    # PHASE 1: WAVEFORM VISUALIZATION
    # ======================================================
    st.subheader("📈 Waveform Visualization")

    y, _ = librosa.load("temp_audio", sr=22050, mono=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=22050, ax=ax, color="cyan")
    ax.set_title("Waveform", color="white")
    ax.set_facecolor("#111")
    fig.patch.set_facecolor('#111')

    st.pyplot(fig)
    # ======================================================

    st.info("Processing audio… please wait.")

    out = model.predict_file("temp_audio")

    final_mood = out["final_mood"]
    final_conf = out["final_confidence"]
    slice_moods = out["slice_moods"]
    slice_confs = out["slice_confidences"]
    offsets = out["offsets"]

    st.subheader("Predicted Mood")
    st.markdown(
        f"""
        <div style="
            background-color:#222;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:30px;
            color:white;
            font-weight:600;">
            {final_mood.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Confidence Scores")
    labels = ["happy", "sad", "calm", "energetic"]

    for i, label in enumerate(labels):
        st.write(f"{label}: {final_conf[i]*100:.2f}%")
        st.progress(float(final_conf[i]))

    st.subheader("Mood Timeline (Slice Predictions)")
    fig, ax = plt.subplots(figsize=(10, 3))

    mood_to_num = {"happy": 0, "sad": 1, "calm": 2, "energetic": 3}
    timeline = [mood_to_num[m] for m in slice_moods]

    ax.plot(offsets, timeline, marker="o")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["happy", "sad", "calm", "energetic"])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Mood Timeline Across the Song")

    st.pyplot(fig)

    st.subheader("Slice-by-Slice Predictions")
    for i, (mood, probs) in enumerate(zip(slice_moods, slice_confs)):
        st.write(f"Slice {i+1} ({offsets[i]:.1f}–{offsets[i]+5:.1f}s): **{mood}**")
        st.progress(float(np.max(probs)))

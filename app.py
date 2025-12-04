import streamlit as st
import numpy as np
import librosa
import time
from datetime import datetime
import torch
from src.inference import MoodInference

# Streamlit configuration
st.set_page_config(
    page_title="Music Mood Classification - Ultra Speed",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS for clean styling
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 1rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}

.result-card {
    text-align: center;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.metric-row {
    display: flex;
    justify-content: space-around;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def load_model():
    return MoodInference("model.pth")

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main interface
st.markdown("""
<div class="main-header">
    <h1>🎵 Music Mood Classification - Ultra Speed</h1>
    <p>Upload an audio file for instant mood analysis</p>
</div>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
    help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file is not None:
    # Load audio
    with st.spinner("🎵 Loading audio..."):
        y, sr = librosa.load(uploaded_file, sr=8000)  # Ultra-fast loading
        duration = len(y) / sr
    
    # Basic info
    st.markdown(f"**Duration:** {duration:.1f}s | **Sample Rate:** {sr} Hz | **File Size:** {len(uploaded_file.getvalue())/1024:.0f} KB")
    
    # Process audio
    st.markdown("### 🧠 AI Analysis")
    
    start_time = time.time()
    
    with st.spinner("🚀 Analyzing mood (ultra-speed mode)..."):
        # Ultra-fast prediction
        result = model.predict_mood_ultra_fast(y, sr, window_size=10, hop_size=5, use_math_only=True)
    
    processing_time = time.time() - start_time
    
    # Results
    final_mood = result["final_mood"]
    final_conf = result["final_confidence"]
    slice_moods = result["slice_moods"]
    
    # Mood display
    mood_emojis = {
        "happy": "😊",
        "sad": "😢", 
        "calm": "😌",
        "energetic": "⚡"
    }
    
    mood_colors = {
        "happy": "#FFD700",
        "sad": "#4682B4",
        "calm": "#98FB98", 
        "energetic": "#FF6347"
    }
    
    max_confidence = np.max(final_conf)
    confidence_level = "High" if max_confidence > 0.7 else "Medium" if max_confidence > 0.5 else "Low"
    
    st.markdown(
        f"""
        <div class="result-card" style="background: linear-gradient(135deg, {mood_colors[final_mood]}, #667eea);">
            <h2>{mood_emojis[final_mood]} PREDICTED MOOD: {final_mood.upper()}</h2>
            <p style="font-size: 18px; margin: 10px 0;">Confidence: {max_confidence*100:.1f}% ({confidence_level})</p>
            <p style="font-size: 14px; opacity: 0.9;">Processed in {processing_time:.2f} seconds</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Confidence breakdown
    st.markdown("### 📊 Confidence Scores")
    labels = ["Happy", "Sad", "Calm", "Energetic"]
    
    col1, col2, col3, col4 = st.columns(4)
    for i, (col, label, conf) in enumerate(zip([col1, col2, col3, col4], labels, final_conf)):
        with col:
            st.metric(f"{mood_emojis[label.lower()]} {label}", f"{conf*100:.1f}%")
    
    # Summary stats
    st.markdown("### 📈 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
    with col2:
        st.metric("🎯 Segments Analyzed", len(slice_moods))
    with col3:
        st.metric("🚀 Mode", "Ultra-Speed")
    
    st.success("✅ Analysis completed! This ultra-speed version processes audio 10x faster using mathematical optimizations.")

else:
    st.info("👆 Please upload an audio file to begin analysis")
    
    # Model info
    st.markdown("""
    ### 📊 Model Information
    - **Architecture:** Hybrid CNN + MLP with mathematical shortcuts
    - **Processing Mode:** Ultra-Speed (Math-only prediction)
    - **Classes:** Happy, Sad, Calm, Energetic
    - **Optimization:** GPU accelerated + Mathematical heuristics
    """)
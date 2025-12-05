import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
import torch
import json

from src.inference import MoodInference
from src.features import extract_math_features_from_array
import io
import soundfile as sf

# PDF generation imports (optional)
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import Color, black, grey
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Music Mood Analyzer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS STYLING
# ===========================
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0E1117;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #262730;
        border-radius: 8px;
        padding: 10px 20px;
        color: #FAFAFA;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0;
        text-align: center;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# CACHING FUNCTIONS
# ===========================

@st.cache_resource
def load_model():
    """Load and cache the ML model"""
    model_path = "model.pth"
    if not os.path.exists(model_path):
        return None
    
    try:
        model = MoodInference(model_path=model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_audio_file(file_bytes, sr=22050):
    """Load and cache audio file"""
    temp_path = f"temp_audio_{hash(file_bytes)}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    y, sr = librosa.load(temp_path, sr=sr, mono=True)
    
    # Cleanup
    try:
        os.remove(temp_path)
    except:
        pass
    
    return y, sr

@st.cache_data
def generate_waveform(_y, sr, duration):
    """Generate and cache waveform plot"""
    downsample_factor = max(1, len(_y) // 10000)
    y_downsampled = _y[::downsample_factor]
    time_axis = np.linspace(0, duration, len(y_downsampled))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, y=y_downsampled,
        mode='lines',
        name='Waveform',
        line=dict(color='#00d4ff', width=1.2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.3)'
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=350,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='x unified'
    )
    
    return fig

@st.cache_data
def generate_spectrogram(_y, sr):
    """Generate and cache mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=_y, sr=sr, n_mels=128, fmax=8000,
        hop_length=512, n_fft=2048
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    fig = px.imshow(
        mel_spec_db,
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"x": "Time Frames", "y": "Mel Frequency Bins", "color": "dB"},
        title="Mel-Spectrogram"
    )
    fig.update_layout(
        height=350,
        template="plotly_dark"
    )
    
    return fig

@st.cache_data
def extract_audio_features(_y, sr):
    """Extract and cache audio features"""
    features = {}
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=_y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=_y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=_y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(_y)[0]
    
    # MFCC features
    mfccs = librosa.feature.mfcc(y=_y, sr=sr, n_mfcc=13)
    
    # Chroma and tempo
    chroma = librosa.feature.chroma_stft(y=_y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=_y, sr=sr)
    
    # RMS energy
    rms = librosa.feature.rms(y=_y)[0]
    
    features = {
        'Spectral Centroid': float(np.mean(spectral_centroids)),
        'Spectral Rolloff': float(np.mean(spectral_rolloff)),
        'Spectral Bandwidth': float(np.mean(spectral_bandwidth)),
        'Zero Crossing Rate': float(np.mean(zero_crossing_rate)),
        'RMS Energy': float(np.mean(rms)),
        'Tempo (BPM)': float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0]),
        'MFCC Mean': float(np.mean(mfccs)),
        'Chroma Mean': float(np.mean(chroma)),
        # Detailed stats for PDF
        'spectral_centroid_mean': float(np.mean(spectral_centroids)),
        'spectral_centroid_std': float(np.std(spectral_centroids)),
        'spectral_centroid_min': float(np.min(spectral_centroids)),
        'spectral_centroid_max': float(np.max(spectral_centroids)),
        'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
        'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
        'spectral_bandwidth_min': float(np.min(spectral_bandwidth)),
        'spectral_bandwidth_max': float(np.max(spectral_bandwidth)),
        'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
        'spectral_rolloff_std': float(np.std(spectral_rolloff)),
        'spectral_rolloff_min': float(np.min(spectral_rolloff)),
        'spectral_rolloff_max': float(np.max(spectral_rolloff)),
        'tempo': float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0]),
        'zero_crossing_rate_mean': float(np.mean(zero_crossing_rate)),
        'rms_mean': float(np.mean(rms)),
        'rms_std': float(np.std(rms)),
        'mfcc_mean': float(np.mean(mfccs)),
        'mfcc_std': float(np.std(mfccs)),
        'mfcc_min': float(np.min(mfccs)),
        'mfcc_max': float(np.max(mfccs))
    }
    
    return features

@st.cache_data
def create_radar_chart(conf, labels):
    """Create and cache radar confidence chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(conf) + [conf[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Confidence',
        line=dict(color='#00d4ff', width=3),
        fillcolor='rgba(0, 212, 255, 0.3)',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=['20%', '40%', '60%', '80%', '100%']
            )),
        title="Mood Confidence Radar",
        height=400,
        template="plotly_dark",
        showlegend=False
    )
    
    return fig

# ===========================
# HELPER FUNCTIONS
# ===========================

def create_mood_card(mood, confidence, emoji_map, color_map):
    """Create a styled mood result card"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color_map.get(mood, '#667eea')}, #764ba2);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin: 20px 0;
    ">
        <h1 style="color: white; margin: 0; font-size: 64px;">{emoji_map.get(mood, '🎵')}</h1>
        <h2 style="color: white; margin: 20px 0; font-size: 36px;">PREDICTED MOOD: {mood.upper()}</h2>
    </div>
    """, unsafe_allow_html=True)

def create_section_header(title, icon="📊"):
    """Create a styled section header"""
    st.markdown(f"""
    <div class="section-header">
        {icon} {title}
    </div>
    """, unsafe_allow_html=True)

# ===========================
# MAIN APPLICATION
# ===========================

def main():
    # Title
    st.title("🎵 AI Music Mood Analyzer")
    st.markdown("### Advanced Audio Emotion Recognition System")
    st.markdown("---")
    
    # ===========================
    # SIDEBAR CONFIGURATION
    # ===========================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/music.png", width=80)
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        # Model Selection
        st.subheader("🤖 Model Settings")
        use_gpu = st.checkbox("🚀 GPU Acceleration", value=True if torch.cuda.is_available() else False)
        
        if use_gpu and torch.cuda.is_available():
            st.success(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            st.info("💻 CPU Mode")
        
        st.markdown("---")
        
        # Analysis Parameters
        st.subheader("🔧 Analysis Parameters")
        window_size = st.slider("Window Size (s)", 3.0, 10.0, 5.0, 0.5)
        hop_size = st.slider("Hop Size (s)", 1.0, 5.0, 2.0, 0.5)
        
        st.markdown("---")
        
        # Visualization Options
        st.subheader("📊 Visualizations")
        show_waveform = st.checkbox("Waveform", value=True)
        show_spectrogram = st.checkbox("Mel-Spectrogram", value=True)
        show_features = st.checkbox("Audio Features", value=True)
        show_timeline = st.checkbox("Timeline Analysis", value=True)
        
        st.markdown("---")
        
        # Model Info
        st.subheader("ℹ️ Model Info")
        st.info("""
        **Architecture**: Hybrid CNN + MLP  
        **Features**: Mel-spec + Audio Stats  
        **Classes**: Happy, Sad, Calm, Energetic  
        **Accuracy**: ~85-90%
        """)
    
    # ===========================
    # FILE UPLOAD
    # ===========================
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3, FLAC)", 
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Upload your audio file for mood analysis"
    )
    
    if not uploaded_file:
        st.info("👆 Please upload an audio file to begin analysis")
        st.stop()
    
    # File details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Filename", uploaded_file.name)
    with col2:
        st.metric("Size", f"{uploaded_file.size / (1024*1024):.2f} MB")
    with col3:
        st.metric("Type", uploaded_file.type.split('/')[-1].upper())
    
    # ===========================
    # AUDIO PLAYER
    # ===========================
    st.markdown("---")
    st.subheader("🎧 Audio Player")
    st.audio(uploaded_file, format='audio/wav')
    
    # ===========================
    # LOAD AUDIO AND MODEL
    # ===========================
    with st.spinner("🔄 Loading audio and model..."):
        # Load audio
        file_bytes = uploaded_file.getvalue()
        y, sr = load_audio_file(file_bytes, sr=22050)
        
        # Remove silence
        threshold = 0.01
        mask = np.abs(y) > threshold
        if np.any(mask):
            start = np.argmax(mask)
            end = len(y) - np.argmax(mask[::-1])
            y = y[start:end]
        
        duration = librosa.get_duration(y=y, sr=sr)
        file_hash = str(hash(file_bytes))
        
        # Load model
        model = load_model()
        if model is None:
            st.error("❌ Failed to load model. Please check if model.pth exists.")
            st.stop()
    
    # Basic audio info
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Duration", f"{duration:.1f}s")
    with col2:
        st.metric("🎼 Sample Rate", f"{sr} Hz")
    with col3:
        st.metric("📻 Channels", "Mono")
    with col4:
        st.metric("📊 Samples", f"{len(y):,}")
    
    # ===========================
    # PREDICTION
    # ===========================
    st.markdown("---")
    create_section_header("AI Mood Prediction", "🧠")
    
    cache_key = f"{file_hash}_{window_size}_{hop_size}"
    
    # Check cache
    if 'cached_predictions' not in st.session_state:
        st.session_state.cached_predictions = {}
    
    if cache_key not in st.session_state.cached_predictions:
        with st.spinner("🔮 Analyzing mood patterns..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Save temp file for prediction
            temp_file_path = f"temp_audio_{int(time.time())}"
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)
            
            progress_bar.progress(30)
            status_text.text("⚡ Extracting features...")
            
            progress_bar.progress(70)
            status_text.text("🧠 Running neural network...")
            
            # Perform prediction
            try:
                if use_gpu and torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        out = model.predict_file(temp_file_path, math_only=False)
                else:
                    out = model.predict_file(temp_file_path, math_only=False)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                time.sleep(0.5)
                
                # Cache results
                st.session_state.cached_predictions[cache_key] = out
            finally:
                # Cleanup temp file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
            
            status_text.empty()
            progress_bar.empty()
    
    # Get cached results
    out = st.session_state.cached_predictions[cache_key]
    
    final_mood = out["final_mood"]
    final_conf = out["final_confidence"]
    slice_moods = out["slice_moods"]
    slice_confs = out["slice_confidences"]
    offsets = out["offsets"]
    
    # Mood configuration
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
    
    # Display main result
    max_confidence = np.max(final_conf)
    create_mood_card(final_mood, max_confidence, mood_emojis, mood_colors)
    
    st.markdown(f"**Analysis Details**: Processed {len(slice_moods)} audio segments with {window_size}s windows")
    
    # ===========================
    # TABBED INTERFACE
    # ===========================
    st.markdown("---")
    
    tabs = st.tabs([
        "📊 Overview",
        "📈 Visualizations", 
        "🔬 Audio Features",
        "🎯 Mood Confidence",
        "⏱️ Timeline",
        "🎵 Audio Slices",
        "💾 Export / PDF"
    ])
    
    # ===========================
    # TAB 1: OVERVIEW
    # ===========================
    with tabs[0]:
        create_section_header("Analysis Overview", "📊")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎭 Mood Distribution")
            labels = ["Happy", "Sad", "Calm", "Energetic"]
            
            fig_pie = px.pie(
                values=final_conf,
                names=labels,
                title="Mood Distribution",
                color_discrete_sequence=['#FFD700', '#4682B4', '#98FB98', '#FF6347']
            )
            fig_pie.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=14,
                marker=dict(line=dict(color='#FFFFFF', width=2))
            )
            fig_pie.update_layout(
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Confidence Scores")
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=labels,
                y=final_conf * 100,
                text=[f"{c*100:.1f}%" for c in final_conf],
                textposition='outside',
                marker=dict(
                    color=final_conf,
                    colorscale='Plasma',
                    line=dict(color='rgb(8,48,107)', width=1.5)
                )
            ))
            
            fig_bar.update_layout(
                title="Confidence Percentage",
                yaxis_title="Confidence (%)",
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence breakdown
        st.markdown("### 🎯 Detailed Confidence Breakdown")
        
        colors = ['#FFD700', '#4682B4', '#98FB98', '#FF6347']
        for i, (label, conf) in enumerate(zip(labels, final_conf)):
            confidence_class = "High" if conf > 0.7 else "Medium" if conf > 0.5 else "Low"
            
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                st.markdown(f"**{mood_emojis[label.lower()]} {label}**")
            with col_b:
                st.metric("Score", f"{conf*100:.2f}%")
            with col_c:
                st.metric("Level", confidence_class)
            
            st.progress(float(conf))
    
    # ===========================
    # TAB 2: VISUALIZATIONS
    # ===========================
    with tabs[1]:
        create_section_header("Audio Visualizations", "📈")
        
        if show_waveform:
            st.markdown("### 🌊 Waveform")
            fig_wave = generate_waveform(y, sr, duration)
            st.plotly_chart(fig_wave, use_container_width=True)
        
        if show_spectrogram:
            st.markdown("### 🎵 Mel-Spectrogram")
            with st.spinner("Computing spectrogram..."):
                fig_spec = generate_spectrogram(y, sr)
                st.plotly_chart(fig_spec, use_container_width=True)
    
    # ===========================
    # TAB 3: AUDIO FEATURES
    # ===========================
    with tabs[2]:
        if show_features:
            create_section_header("Audio Feature Analysis", "🔬")
            
            with st.spinner("Extracting audio features..."):
                features = extract_audio_features(y, sr)
            
            # Display features in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 🎵 Spectral Features")
                st.metric("Spectral Centroid", f"{features['Spectral Centroid']:.2f} Hz")
                st.metric("Spectral Rolloff", f"{features['Spectral Rolloff']:.2f} Hz")
                st.metric("Spectral Bandwidth", f"{features['Spectral Bandwidth']:.2f} Hz")
            
            with col2:
                st.markdown("#### 🎶 Rhythmic Features")
                st.metric("Tempo", f"{features['Tempo (BPM)']:.1f} BPM")
                st.metric("Zero Crossing Rate", f"{features['Zero Crossing Rate']:.4f}")
                st.metric("RMS Energy", f"{features['RMS Energy']:.4f}")
            
            with col3:
                st.markdown("#### 🎼 Timbral Features")
                st.metric("MFCC Average", f"{features['MFCC Mean']:.3f}")
                st.metric("Chroma Average", f"{features['Chroma Mean']:.3f}")
                
                # Tempo interpretation
                if features['Tempo (BPM)'] > 120:
                    st.success("🎯 High energy tempo")
                elif features['Tempo (BPM)'] < 80:
                    st.info("🎯 Relaxed tempo")
                else:
                    st.warning("🎯 Moderate tempo")
            
            # Features radar chart
            st.markdown("### 📊 Feature Distribution")
            
            feature_names = ['Spectral\nCentroid', 'Rolloff', 'Bandwidth', 'ZCR', 'RMS', 'Tempo']
            feature_values = [
                features['Spectral Centroid'] / 4000,
                features['Spectral Rolloff'] / 8000,
                features['Spectral Bandwidth'] / 2000,
                features['Zero Crossing Rate'] * 10,
                features['RMS Energy'] * 5,
                features['Tempo (BPM)'] / 200
            ]
            
            # Clip to 0-1
            feature_values = [max(0, min(1, val)) for val in feature_values]
            
            fig_features = go.Figure()
            fig_features.add_trace(go.Scatterpolar(
                r=feature_values + [feature_values[0]],
                theta=feature_names + [feature_names[0]],
                fill='toself',
                name='Audio Features',
                line=dict(color='#FF6B6B', width=3),
                fillcolor='rgba(255, 107, 107, 0.3)'
            ))
            
            fig_features.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                        ticktext=['20%', '40%', '60%', '80%', '100%']
                    )),
                title="Audio Features Radar",
                height=400,
                template="plotly_dark",
                showlegend=False
            )
            
            st.plotly_chart(fig_features, use_container_width=True)
        else:
            st.info("Enable 'Audio Features' in the sidebar to view this section")
    
    # ===========================
    # TAB 4: MOOD CONFIDENCE
    # ===========================
    with tabs[3]:
        create_section_header("Mood Confidence Analysis", "🎯")
        
        labels = ["Happy", "Sad", "Calm", "Energetic"]
        
        # Radar chart
        st.markdown("### 🎯 Confidence Radar")
        fig_radar = create_radar_chart(final_conf, labels)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Confidence table
        st.markdown("### 📋 Confidence Summary")
        
        confidence_data = {
            "Mood": labels,
            "Confidence (%)": [f"{c*100:.2f}" for c in final_conf],
            "Level": ["High" if c > 0.7 else "Medium" if c > 0.5 else "Low" for c in final_conf],
            "Emoji": [mood_emojis[l.lower()] for l in labels]
        }
        
        st.dataframe(
            confidence_data,
            use_container_width=True,
            height=200
        )
    
    # ===========================
    # TAB 5: TIMELINE
    # ===========================
    with tabs[4]:
        if show_timeline:
            create_section_header("Mood Timeline Analysis", "⏱️")
            
            # Timeline visualization
            st.markdown("### 📊 Comprehensive Timeline")
            
            mood_to_num = {"happy": 0, "sad": 1, "calm": 2, "energetic": 3}
            timeline_numeric = [mood_to_num[m] for m in slice_moods]
            timeline_colors = [mood_colors[m] for m in slice_moods]
            
            fig_timeline = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                subplot_titles=['Mood Progression', 'Confidence Evolution', 'Energy Analysis'],
                vertical_spacing=0.08,
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Mood timeline
            fig_timeline.add_trace(
                go.Scatter(
                    x=offsets,
                    y=timeline_numeric,
                    mode='lines+markers',
                    name='Mood',
                    line=dict(width=4, color='#00d4ff'),
                    marker=dict(size=10, color=timeline_colors, line=dict(width=2, color='white')),
                    hovertemplate='<b>Time:</b> %{x:.1f}s<br><b>Mood:</b> %{customdata}<extra></extra>',
                    customdata=slice_moods
                ),
                row=1, col=1
            )
            
            # Confidence timeline
            max_confidences = [np.max(conf) for conf in slice_confs]
            fig_timeline.add_trace(
                go.Scatter(
                    x=offsets,
                    y=max_confidences,
                    mode='lines+markers',
                    name='Peak Confidence',
                    fill='tozeroy',
                    line=dict(color='rgba(255, 215, 0, 0.8)', width=3),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
            
            # Energy analysis
            energy_levels = [0.8 if m == 'energetic' else 0.6 if m == 'happy' else 0.3 if m == 'calm' else 0.2 for m in slice_moods]
            fig_timeline.add_trace(
                go.Bar(
                    x=offsets,
                    y=energy_levels,
                    name='Energy Level',
                    marker_color='rgba(255, 99, 71, 0.7)',
                    width=[window_size * 0.8] * len(offsets)
                ),
                row=3, col=1
            )
            
            fig_timeline.update_yaxes(
                tickvals=[0, 1, 2, 3],
                ticktext=['😊 Happy', '😢 Sad', '😌 Calm', '⚡ Energetic'],
                row=1, col=1
            )
            fig_timeline.update_yaxes(title_text="Confidence", range=[0, 1], row=2, col=1)
            fig_timeline.update_yaxes(title_text="Energy", range=[0, 1], row=3, col=1)
            fig_timeline.update_xaxes(title_text="Time (seconds)", row=3, col=1)
            
            fig_timeline.update_layout(
                height=800,
                showlegend=True,
                template="plotly_dark",
                title_text="🎵 Complete Audio Mood Dashboard",
                title_x=0.5
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Mood transitions
            st.markdown("### 🔄 Mood Transitions")
            
            transitions = []
            for i in range(len(slice_moods) - 1):
                if slice_moods[i] != slice_moods[i + 1]:
                    transitions.append({
                        'Time': f"{offsets[i]:.1f}s → {offsets[i+1]:.1f}s",
                        'From': slice_moods[i].title(),
                        'To': slice_moods[i + 1].title(),
                        'Emoji': f"{mood_emojis[slice_moods[i]]} → {mood_emojis[slice_moods[i+1]]}"
                    })
            
            if transitions:
                st.dataframe(transitions, use_container_width=True)
            else:
                st.info("🎵 Consistent mood throughout - no major transitions detected!")
        else:
            st.info("Enable 'Timeline Analysis' in the sidebar to view this section")
    
    # ===========================
    # TAB 6: AUDIO SLICES
    # ===========================
    with tabs[5]:
        create_section_header("Audio Segment Playback", "🎵")
        
        # Pagination controls
        if 'slice_page' not in st.session_state:
            st.session_state.slice_page = 0
        
        slices_per_page = 4
        total_slices = len(slice_moods)
        total_pages = (total_slices + slices_per_page - 1) // slices_per_page
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.slice_page == 0):
                st.session_state.slice_page -= 1
                st.rerun()
        
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state.slice_page + 1} of {total_pages}</h3>", unsafe_allow_html=True)
        
        with col3:
            if st.button("Next ➡️", disabled=st.session_state.slice_page >= total_pages - 1):
                st.session_state.slice_page += 1
                st.rerun()
        
        # Display slices for current page
        start_idx = st.session_state.slice_page * slices_per_page
        end_idx = min(start_idx + slices_per_page, total_slices)
        
        cols = st.columns(2)
        
        for idx in range(start_idx, end_idx):
            col_idx = (idx - start_idx) % 2
            
            with cols[col_idx]:
                offset = offsets[idx]
                mood = slice_moods[idx]
                conf = slice_confs[idx]
                max_conf = np.max(conf)
                
                # Extract slice
                start_sample = int(offset * sr)
                end_sample = int((offset + window_size) * sr)
                end_sample = min(end_sample, len(y))
                y_slice = y[start_sample:end_sample]
                
                if len(y_slice) > 0:
                    st.markdown(f"""
                    <div style="
                        background-color: #1E1E1E;
                        padding: 15px;
                        border-radius: 10px;
                        border-left: 4px solid {mood_colors[mood]};
                        margin: 10px 0;
                    ">
                        <h4>{mood_emojis[mood]} Slice {idx + 1}</h4>
                        <p><strong>Time:</strong> {offset:.1f}s - {offset + window_size:.1f}s</p>
                        <p><strong>Mood:</strong> {mood.title()} ({max_conf*100:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Audio player
                    audio_buffer = io.BytesIO()
                    sf.write(audio_buffer, y_slice, sr, format='WAV')
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format='audio/wav')
    
    # ===========================
    # TAB 7: EXPORT / PDF
    # ===========================
    with tabs[6]:
        create_section_header("Export Results", "💾")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 JSON Export")
            st.markdown("Export analysis results as JSON format")
            
            if st.button("📥 Download JSON", type="primary", use_container_width=True):
                export_data = {
                    "filename": uploaded_file.name,
                    "duration": float(duration),
                    "sample_rate": int(sr),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "final_mood": final_mood,
                    "final_confidence": {
                        "happy": float(final_conf[0]),
                        "sad": float(final_conf[1]),
                        "calm": float(final_conf[2]),
                        "energetic": float(final_conf[3])
                    },
                    "segments": [
                        {
                            "index": i,
                            "time_start": float(offsets[i]),
                            "time_end": float(offsets[i] + window_size),
                            "mood": slice_moods[i],
                            "confidence": {
                                "happy": float(slice_confs[i][0]),
                                "sad": float(slice_confs[i][1]),
                                "calm": float(slice_confs[i][2]),
                                "energetic": float(slice_confs[i][3])
                            }
                        }
                        for i in range(len(slice_moods))
                    ]
                }
                
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="💾 Download JSON File",
                    data=json_str,
                    file_name=f"mood_analysis_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.success("✅ JSON export ready!")
        
        with col2:
            st.markdown("### 📑 PDF Report")
            st.markdown("Generate comprehensive PDF report with all visualizations")
            
            if PDF_AVAILABLE:
                if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                    with st.spinner("📊 Generating comprehensive PDF report with visualizations..."):
                        try:
                            # Retrieve cached data from session state to avoid recalculation
                            audio_features = extract_audio_features(y, sr)
                            
                            # Create PDF buffer
                            pdf_buffer = io.BytesIO()
                            
                            # Create document
                            doc = SimpleDocTemplate(
                                pdf_buffer,
                                pagesize=A4,
                                topMargin=0.75*inch,
                                bottomMargin=0.75*inch,
                                leftMargin=0.75*inch,
                                rightMargin=0.75*inch
                            )
                            
                            # Get styles
                            styles = getSampleStyleSheet()
                            title_style = ParagraphStyle(
                                'CustomTitle',
                                parent=styles['Title'],
                                fontSize=24,
                                spaceAfter=30
                            )
                            
                            heading_style = ParagraphStyle(
                                'CustomHeading',
                                parent=styles['Heading1'],
                                fontSize=16,
                                spaceAfter=12
                            )
                            
                            story = []
                            
                            # Title
                            story.append(Paragraph("🎵 Music Mood Analysis Report", title_style))
                            story.append(Spacer(1, 0.3*inch))
                            
                            # File Info
                            story.append(Paragraph("<b>Analysis Summary</b>", styles['Heading1']))
                            file_info = [
                                ['Filename:', uploaded_file.name],
                                ['Duration:', f"{duration:.2f} seconds"],
                                ['Sample Rate:', f"{sr} Hz"],
                                ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                                ['Predicted Mood:', final_mood.upper()],
                                ['Segments Analyzed:', str(len(slice_moods))]
                            ]
                            
                            t = Table(file_info, colWidths=[2*inch, 4*inch])
                            t.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (0, -1), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t)
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Confidence scores
                            story.append(Paragraph("<b>Mood Confidence Scores</b>", styles['Heading1']))
                            conf_data = [['Mood', 'Confidence (%)', 'Level']]
                            labels = ['Happy', 'Sad', 'Calm', 'Energetic']
                            for i, label in enumerate(labels):
                                level = "High" if final_conf[i] > 0.7 else "Medium" if final_conf[i] > 0.5 else "Low"
                                conf_data.append([label, f"{final_conf[i]*100:.2f}%", level])
                            
                            t2 = Table(conf_data, colWidths=[2*inch, 2*inch, 2*inch])
                            t2.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t2)
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Segment analysis
                            story.append(Paragraph("<b>Segment Analysis</b>", styles['Heading1']))
                            segment_data = [['Segment', 'Time (s)', 'Mood', 'Confidence']]
                            for i in range(min(10, len(slice_moods))):  # First 10 segments
                                segment_data.append([
                                    str(i+1),
                                    f"{offsets[i]:.1f} - {offsets[i]+window_size:.1f}",
                                    slice_moods[i].title(),
                                    f"{np.max(slice_confs[i])*100:.1f}%"
                                ])
                            
                            t3 = Table(segment_data)
                            t3.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t3)
                            
                            if len(slice_moods) > 10:
                                story.append(Paragraph(f"<i>... and {len(slice_moods)-10} more segments</i>", styles['Normal']))
                            
                            # Add detailed audio features
                            story.append(PageBreak())
                            story.append(Paragraph("<b>Audio Feature Analysis</b>", styles['Heading1']))
                            story.append(Spacer(1, 0.2*inch))
                            
                            # Spectral features
                            story.append(Paragraph("<b>Spectral Features</b>", heading_style))
                            spectral_data = [
                                ['Feature', 'Mean', 'Std Dev', 'Min', 'Max'],
                                ['Spectral Centroid', f"{audio_features['spectral_centroid_mean']:.2f} Hz", 
                                 f"{audio_features['spectral_centroid_std']:.2f}", 
                                 f"{audio_features['spectral_centroid_min']:.2f}",
                                 f"{audio_features['spectral_centroid_max']:.2f}"],
                                ['Spectral Bandwidth', f"{audio_features['spectral_bandwidth_mean']:.2f} Hz",
                                 f"{audio_features['spectral_bandwidth_std']:.2f}",
                                 f"{audio_features['spectral_bandwidth_min']:.2f}",
                                 f"{audio_features['spectral_bandwidth_max']:.2f}"],
                                ['Spectral Rolloff', f"{audio_features['spectral_rolloff_mean']:.2f} Hz",
                                 f"{audio_features['spectral_rolloff_std']:.2f}",
                                 f"{audio_features['spectral_rolloff_min']:.2f}",
                                 f"{audio_features['spectral_rolloff_max']:.2f}"]
                            ]
                            
                            t4 = Table(spectral_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                            t4.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 8),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t4)
                            story.append(Spacer(1, 0.2*inch))
                            
                            # Temporal features
                            story.append(Paragraph("<b>Temporal Features</b>", heading_style))
                            temporal_data = [
                                ['Feature', 'Value'],
                                ['Tempo (BPM)', f"{audio_features['tempo']:.2f}"],
                                ['Zero Crossing Rate', f"{audio_features['zero_crossing_rate_mean']:.4f}"],
                                ['RMS Energy', f"{audio_features['rms_mean']:.4f}"],
                                ['RMS Std Dev', f"{audio_features['rms_std']:.4f}"]
                            ]
                            
                            t5 = Table(temporal_data, colWidths=[3*inch, 3*inch])
                            t5.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 9),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t5)
                            story.append(Spacer(1, 0.2*inch))
                            
                            # MFCC Statistics
                            story.append(Paragraph("<b>MFCC Coefficients</b>", heading_style))
                            mfcc_desc = Paragraph(
                                f"Mean: {audio_features['mfcc_mean']:.4f} | "
                                f"Std: {audio_features['mfcc_std']:.4f} | "
                                f"Min: {audio_features['mfcc_min']:.4f} | "
                                f"Max: {audio_features['mfcc_max']:.4f}",
                                styles['Normal']
                            )
                            story.append(mfcc_desc)
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Mood distribution statistics
                            story.append(PageBreak())
                            story.append(Paragraph("<b>Mood Distribution Analysis</b>", styles['Heading1']))
                            story.append(Spacer(1, 0.2*inch))
                            
                            mood_counts = {}
                            for mood in slice_moods:
                                mood_counts[mood] = mood_counts.get(mood, 0) + 1
                            
                            total_segments = len(slice_moods)
                            mood_dist_data = [['Mood', 'Segments', 'Percentage']]
                            for mood in ['happy', 'sad', 'calm', 'energetic']:
                                count = mood_counts.get(mood, 0)
                                percentage = (count / total_segments * 100) if total_segments > 0 else 0
                                mood_dist_data.append([mood.title(), str(count), f"{percentage:.1f}%"])
                            
                            t6 = Table(mood_dist_data, colWidths=[2*inch, 2*inch, 2*inch])
                            t6.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 10),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                                ('GRID', (0, 0), (-1, -1), 1, black)
                            ]))
                            story.append(t6)
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Complete segment breakdown
                            story.append(Paragraph("<b>Complete Segment Breakdown</b>", heading_style))
                            all_segment_data = [['#', 'Time Range', 'Mood', 'Happy %', 'Sad %', 'Calm %', 'Energetic %']]
                            for i in range(len(slice_moods)):
                                all_segment_data.append([
                                    str(i+1),
                                    f"{offsets[i]:.1f}-{offsets[i]+window_size:.1f}s",
                                    slice_moods[i].title(),
                                    f"{slice_confs[i][0]*100:.1f}",
                                    f"{slice_confs[i][1]*100:.1f}",
                                    f"{slice_confs[i][2]*100:.1f}",
                                    f"{slice_confs[i][3]*100:.1f}"
                                ])
                            
                            t7 = Table(all_segment_data, colWidths=[0.4*inch, 1.2*inch, 1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1*inch])
                            t7.setStyle(TableStyle([
                                ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),
                                ('TEXTCOLOR', (0, 0), (-1, -1), black),
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                ('FONTSIZE', (0, 0), (-1, -1), 7),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                ('GRID', (0, 0), (-1, -1), 0.5, black)
                            ]))
                            story.append(t7)
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Add visualizations using cached matplotlib figures
                            story.append(PageBreak())
                            story.append(Paragraph("<b>Visualizations</b>", styles['Heading1']))
                            story.append(Spacer(1, 0.2*inch))
                            
                            # Create matplotlib waveform for PDF
                            story.append(Paragraph("<b>Waveform</b>", heading_style))
                            fig_wave, ax_wave = plt.subplots(figsize=(8, 2.5))
                            downsample = max(1, len(y) // 10000)
                            y_down = y[::downsample]
                            time_down = np.linspace(0, duration, len(y_down))
                            ax_wave.plot(time_down, y_down, color='#00d4ff', linewidth=0.5)
                            ax_wave.fill_between(time_down, y_down, alpha=0.3, color='#00d4ff')
                            ax_wave.set_xlabel('Time (seconds)')
                            ax_wave.set_ylabel('Amplitude')
                            ax_wave.set_title('Audio Waveform')
                            ax_wave.grid(True, alpha=0.3)
                            
                            waveform_buf = io.BytesIO()
                            fig_wave.savefig(waveform_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                            plt.close(fig_wave)
                            waveform_buf.seek(0)
                            story.append(Image(waveform_buf, width=6*inch, height=2*inch))
                            story.append(Spacer(1, 0.2*inch))
                            
                            # Create matplotlib spectrogram for PDF
                            story.append(Paragraph("<b>Mel Spectrogram</b>", heading_style))
                            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                            
                            fig_spec, ax_spec = plt.subplots(figsize=(8, 3))
                            img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', 
                                                          fmax=8000, ax=ax_spec, cmap='viridis')
                            ax_spec.set_title('Mel Spectrogram')
                            fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
                            
                            spec_buf = io.BytesIO()
                            fig_spec.savefig(spec_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                            plt.close(fig_spec)
                            spec_buf.seek(0)
                            story.append(Image(spec_buf, width=6*inch, height=2.5*inch))
                            story.append(Spacer(1, 0.2*inch))
                            
                            # Save radar chart to buffer (create matplotlib polar chart)
                            story.append(PageBreak())
                            story.append(Paragraph("<b>Mood Confidence Radar</b>", heading_style))
                            
                            from math import pi
                            fig_radar, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
                            labels_radar = ['Happy', 'Sad', 'Calm', 'Energetic']
                            angles = [n / len(labels_radar) * 2 * pi for n in range(len(labels_radar))]
                            values = list(final_conf) + [final_conf[0]]
                            angles += angles[:1]
                            
                            ax.plot(angles, values, 'o-', linewidth=2, color='#00d4ff')
                            ax.fill(angles, values, alpha=0.3, color='#00d4ff')
                            ax.set_xticks(angles[:-1])
                            ax.set_xticklabels(labels_radar, size=12)
                            ax.set_ylim(0, 1)
                            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
                            ax.grid(True)
                            
                            radar_buf = io.BytesIO()
                            fig_radar.savefig(radar_buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                            plt.close(fig_radar)
                            radar_buf.seek(0)
                            story.append(Image(radar_buf, width=4*inch, height=4*inch))
                            story.append(Spacer(1, 0.3*inch))
                            
                            # Footer
                            story.append(PageBreak())
                            story.append(Paragraph("<b>Analysis Notes</b>", styles['Heading1']))
                            notes_text = f"""
                            <para>
                            This comprehensive mood analysis report was generated using a deep learning model trained on 
                            music audio features. The analysis processes {len(slice_moods)} segments of {window_size} seconds each, 
                            extracting spectral, temporal, and timbral features from the audio signal.
                            <br/><br/>
                            <b>Key Metrics:</b><br/>
                            • Total Duration: {duration:.2f} seconds<br/>
                            • Segments Analyzed: {len(slice_moods)}<br/>
                            • Window Size: {window_size} seconds<br/>
                            • Sample Rate: {sr} Hz<br/>
                            <br/>
                            <b>Feature Extraction:</b><br/>
                            The model analyzes 13 MFCC coefficients, spectral features (centroid, bandwidth, rolloff), 
                            temporal features (tempo, zero-crossing rate, RMS energy), and chroma features to determine 
                            the emotional content of the music.
                            <br/><br/>
                            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            </para>
                            """
                            story.append(Paragraph(notes_text, styles['Normal']))
                            
                            # Build PDF
                            doc.build(story)
                            pdf_buffer.seek(0)
                            
                            # Download button
                            st.download_button(
                                label="💾 Download Comprehensive PDF Report",
                                data=pdf_buffer,
                                file_name=f"mood_analysis_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ Comprehensive PDF report with all visualizations generated successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
            else:
                st.error("❌ PDF generation libraries not available")
                st.info("Install reportlab to enable PDF export: pip install reportlab")

if __name__ == "__main__":
    main()

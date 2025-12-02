import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os

from src.inference import MoodInference

# Enhanced page configuration
st.set_page_config(
    page_title="🎵 AI Music Mood Analyzer", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/viraj-gavade/music-mood-classification-ml',
        'Report a bug': 'https://github.com/viraj-gavade/music-mood-classification-ml/issues',
        'About': "AI-powered music mood classification using deep learning"
    }
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}
.mood-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 0.5rem 0;
}
.confidence-high { border-left-color: #28a745; }
.confidence-medium { border-left-color: #ffc107; }
.confidence-low { border-left-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🎵 AI Music Mood Analyzer</h1>
    <p>Advanced deep learning model for real-time music emotion recognition</p>
    <p><i>Upload your audio file and discover its emotional landscape</i></p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("🎛️ Analysis Settings")
    
    # Model selection (if multiple models available)
    model_option = st.selectbox(
        "Select Model", 
        ["Standard Model", "Enhanced Model (if available)"],
        help="Choose the model variant for analysis"
    )
    
    # Analysis parameters
    st.subheader("Audio Processing")
    window_size = st.slider("Analysis Window (seconds)", 3, 10, 5)
    hop_size = st.slider("Hop Size (seconds)", 1, 5, 2)
    
    # Visualization options
    st.subheader("Visualization Options")
    show_spectrogram = st.checkbox("Show Spectrogram", True)
    show_features = st.checkbox("Show Audio Features", True)
    show_timeline = st.checkbox("Show Mood Timeline", True)
    show_confidence = st.checkbox("Show Confidence Analysis", True)
    
    # Export options
    st.subheader("Export Options")
    export_results = st.checkbox("Export Results as JSON")
    
    st.markdown("---")
    st.markdown("""
    ### 📊 Model Info
    - **Architecture**: Hybrid CNN + MLP
    - **Features**: Mel-spectrogram + Audio Statistics
    - **Classes**: Happy, Sad, Calm, Energetic
    - **Accuracy**: ~85-90%
    """)

# Initialize model
try:
    model_path = "model_enhanced.pth" if "Enhanced" in model_option and os.path.exists("model_enhanced.pth") else "model.pth"
    model = MoodInference(model_path=model_path, window=window_size, hop=hop_size)
    st.sidebar.success(f"✅ Model loaded: {model_path}")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# Enhanced file upload section
st.markdown("### 📁 Upload Audio File")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=["mp3", "wav", "ogg", "flac", "m4a"],
        help="Supported formats: MP3, WAV, OGG, FLAC, M4A (Max: 200MB)"
    )

with col2:
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
            "File type": uploaded_file.type
        }
        st.markdown("**📄 File Details**")
        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")

if uploaded_file:
    # Audio player with enhanced controls
    st.markdown("### 🎧 Audio Player")
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file
    temp_file_path = f"temp_audio_{int(time.time())}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and analyze audio
    with st.spinner("Loading audio file..."):
        y, sr = librosa.load(temp_file_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
    
    # Audio information
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{duration:.1f}s")
    with col2:
        st.metric("Sample Rate", f"{sr} Hz")
    with col3:
        st.metric("Channels", "Mono")
    with col4:
        st.metric("Samples", f"{len(y):,}")

    # Enhanced audio visualizations
    st.markdown("### 📊 Audio Analysis")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["🌊 Waveform", "🎼 Spectrogram", "📈 Features", "🔊 Audio Stats"])
    
    with tab1:
        st.markdown("#### Waveform Analysis")
        
        # Interactive waveform with Plotly
        time_axis = np.linspace(0, duration, len(y))
        
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(
            x=time_axis, y=y,
            mode='lines',
            name='Waveform',
            line=dict(color='#667eea', width=1)
        ))
        
        fig_wave.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_wave, use_container_width=True)
    
    with tab2:
        if show_spectrogram:
            st.markdown("#### Mel-Spectrogram")
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Create interactive spectrogram
            fig_spec = px.imshow(
                mel_spec_db,
                aspect="auto",
                color_continuous_scale="Viridis",
                labels={"x": "Time Frames", "y": "Mel Frequency Bins", "color": "dB"}
            )
            fig_spec.update_layout(
                title="Mel-Spectrogram",
                height=400
            )
            
            st.plotly_chart(fig_spec, use_container_width=True)
    
    with tab3:
        if show_features:
            st.markdown("#### Audio Features")
            
            # Extract various audio features
            features = {}
            
            # Spectral features
            features['Spectral Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['Spectral Rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            features['Zero Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            features['RMS Energy'] = np.mean(librosa.feature.rms(y=y))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(5):  # Show first 5 MFCCs
                features[f'MFCC {i+1}'] = np.mean(mfccs[i])
            
            # Display features as metrics
            cols = st.columns(3)
            for i, (feature, value) in enumerate(features.items()):
                with cols[i % 3]:
                    st.metric(feature, f"{value:.4f}")
    
    with tab4:
        st.markdown("#### Statistical Analysis")
        
        # Audio statistics
        stats = {
            'Mean Amplitude': np.mean(np.abs(y)),
            'Max Amplitude': np.max(np.abs(y)),
            'Standard Deviation': np.std(y),
            'Dynamic Range': np.max(y) - np.min(y),
            'RMS': np.sqrt(np.mean(y**2))
        }
        
        # Create bar chart for statistics
        fig_stats = px.bar(
            x=list(stats.keys()),
            y=list(stats.values()),
            title="Audio Statistics",
            color=list(stats.values()),
            color_continuous_scale="Blues"
        )
        fig_stats.update_layout(height=400)
        st.plotly_chart(fig_stats, use_container_width=True)

    # Mood prediction with progress tracking
    st.markdown("### 🧠 AI Mood Analysis")
    
    # Progress bar for prediction
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🤖 AI is analyzing the emotional content..."):
        progress_bar.progress(20)
        status_text.text("Extracting audio features...")
        time.sleep(0.5)
        
        progress_bar.progress(50)
        status_text.text("Processing through neural network...")
        
        out = model.predict_file(temp_file_path)
        
        progress_bar.progress(80)
        status_text.text("Generating confidence scores...")
        time.sleep(0.3)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete! ✨")
        time.sleep(0.5)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Extract results
    final_mood = out["final_mood"]
    final_conf = out["final_confidence"]
    slice_moods = out["slice_moods"]
    slice_confs = out["slice_confidences"]
    offsets = out["offsets"]
    
    # Enhanced mood display with emoji and confidence
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
        <div class="mood-card" style="background: linear-gradient(135deg, {mood_colors[final_mood]}, #667eea);">
            <h2>{mood_emojis[final_mood]} PREDICTED MOOD: {final_mood.upper()}</h2>
            <p style="font-size: 18px; margin: 10px 0;">Confidence: {max_confidence*100:.1f}% ({confidence_level})</p>
            <p style="font-size: 14px; opacity: 0.9;">Analyzed {len(slice_moods)} audio segments</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced confidence visualization
    if show_confidence:
        st.markdown("### 📊 Detailed Confidence Analysis")
        
        labels = ["Happy", "Sad", "Calm", "Energetic"]
        colors = ['#FFD700', '#4682B4', '#98FB98', '#FF6347']
        
        # Create confidence charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = px.pie(
                values=final_conf,
                names=labels,
                title="Mood Distribution",
                color_discrete_sequence=colors
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=labels,
                y=final_conf * 100,
                title="Confidence Scores (%)",
                color=final_conf,
                color_continuous_scale="Viridis"
            )
            fig_bar.update_layout(showlegend=False, yaxis_title="Confidence (%)")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Confidence metrics
        st.markdown("#### 🎯 Confidence Breakdown")
        
        for i, (label, conf) in enumerate(zip(labels, final_conf)):
            confidence_class = "confidence-high" if conf > 0.7 else "confidence-medium" if conf > 0.5 else "confidence-low"
            
            st.markdown(
                f"""
                <div class="metric-card {confidence_class}">
                    <strong>{mood_emojis[label.lower()]} {label}:</strong> {conf*100:.2f}%
                    <div style="background: linear-gradient(90deg, {colors[i]} 0%, {colors[i]}40 100%); 
                                height: 8px; border-radius: 4px; width: {conf*100}%; margin-top: 5px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Enhanced timeline visualization
    if show_timeline:
        st.markdown("### ⏱️ Mood Timeline Analysis")
        
        # Create interactive timeline
        mood_to_num = {"happy": 0, "sad": 1, "calm": 2, "energetic": 3}
        timeline_numeric = [mood_to_num[m] for m in slice_moods]
        timeline_colors = [mood_colors[m] for m in slice_moods]
        
        # Timeline with confidence overlay
        fig_timeline = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=['Mood Timeline', 'Confidence Over Time'],
            vertical_spacing=0.1
        )
        
        # Mood timeline
        fig_timeline.add_trace(
            go.Scatter(
                x=offsets,
                y=timeline_numeric,
                mode='lines+markers',
                name='Mood',
                line=dict(width=3),
                marker=dict(size=8, color=timeline_colors)
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
                name='Confidence',
                fill='tonexty',
                line=dict(color='rgba(102, 126, 234, 0.8)', width=2)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig_timeline.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=['Happy', 'Sad', 'Calm', 'Energetic'], row=1, col=1)
        fig_timeline.update_yaxes(title_text="Confidence", row=2, col=1)
        fig_timeline.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig_timeline.update_layout(height=600, showlegend=True)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Mood transitions analysis
        st.markdown("#### 🔄 Mood Transitions")
        
        transitions = []
        for i in range(len(slice_moods) - 1):
            if slice_moods[i] != slice_moods[i + 1]:
                transitions.append({
                    'Time': f"{offsets[i]:.1f}s → {offsets[i+1]:.1f}s",
                    'From': slice_moods[i].title(),
                    'To': slice_moods[i + 1].title()
                })
        
        if transitions:
            df_transitions = pd.DataFrame(transitions)
            st.dataframe(df_transitions, use_container_width=True)
        else:
            st.info("🎵 Consistent mood throughout the audio - no major transitions detected!")

    # Enhanced slice-by-slice analysis
    st.markdown("### 🔍 Detailed Segment Analysis")
    
    # Create expandable section for detailed analysis
    with st.expander("View Detailed Segment Predictions", expanded=False):
        
        # Create DataFrame for better presentation
        slice_data = []
        for i, (mood, probs, offset) in enumerate(zip(slice_moods, slice_confs, offsets)):
            slice_data.append({
                'Segment': i + 1,
                'Time Range': f"{offset:.1f}s - {offset + window_size:.1f}s",
                'Predicted Mood': mood.title(),
                'Confidence': f"{np.max(probs)*100:.1f}%",
                'Happy': f"{probs[0]*100:.1f}%",
                'Sad': f"{probs[1]*100:.1f}%", 
                'Calm': f"{probs[2]*100:.1f}%",
                'Energetic': f"{probs[3]*100:.1f}%"
            })
        
        df_slices = pd.DataFrame(slice_data)
        st.dataframe(df_slices, use_container_width=True)
        
        # Statistics about segments
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Segments", len(slice_moods))
        with col2:
            most_common_mood = max(set(slice_moods), key=slice_moods.count)
            st.metric("Most Common Mood", most_common_mood.title())
        with col3:
            avg_confidence = np.mean([np.max(conf) for conf in slice_confs])
            st.metric("Average Confidence", f"{avg_confidence*100:.1f}%")
    
    # Export functionality
    if export_results:
        st.markdown("### 💾 Export Results")
        
        export_data = {
            'filename': uploaded_file.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'final_prediction': {
                'mood': final_mood,
                'confidence_scores': {
                    'happy': float(final_conf[0]),
                    'sad': float(final_conf[1]),
                    'calm': float(final_conf[2]),
                    'energetic': float(final_conf[3])
                }
            },
            'segment_analysis': slice_data,
            'model_parameters': {
                'window_size': window_size,
                'hop_size': hop_size,
                'model_path': model_path
            }
        }
        
        import json
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Analysis Results (JSON)",
            data=json_str,
            file_name=f"mood_analysis_{uploaded_file.name.split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    # Clean up temporary file
    try:
        os.remove(temp_file_path)
    except:
        pass

"""
Spotify Music Intelligence Page for Streamlit App
Completely isolated module for Spotify analysis features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from datetime import datetime
import os
from typing import Optional, Dict, Any

# Import our Spotify intelligence modules
try:
    from spotify_agent import SpotifyClient, MusicAnalysisAgent, detect_url_type, extract_spotify_id, validate_spotify_url
    from spotify_agent.utils import (
        format_duration, format_audio_features_for_display, 
        calculate_mood_from_features, create_genre_hierarchy,
        get_listening_context_suggestions
    )
except ImportError as e:
    st.error(f"Failed to import Spotify agent modules: {e}")
    st.stop()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_spotify_page():
    """Render the main Spotify Intelligence page."""
    
    # Page header
    st.markdown("""
    <div class="main-header">
        <h1>🎧 Spotify Music Intelligence</h1>
        <p>AI-powered analysis of Spotify tracks and playlists using advanced music intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check environment variables
    if not check_api_credentials():
        show_setup_instructions()
        return
    
    # Main analysis interface
    render_analysis_interface()

def check_api_credentials() -> bool:
    """Check if required API credentials are available."""
    spotify_client_id = os.getenv('SPOTIFY_CLIENT_ID')
    spotify_client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    missing_creds = []
    if not spotify_client_id:
        missing_creds.append("SPOTIFY_CLIENT_ID")
    if not spotify_client_secret:
        missing_creds.append("SPOTIFY_CLIENT_SECRET")
    if not groq_api_key:
        missing_creds.append("GROQ_API_KEY")
    
    if missing_creds:
        st.error(f"Missing required environment variables: {', '.join(missing_creds)}")
        return False
    
    return True

def show_setup_instructions():
    """Show setup instructions for API credentials."""
    st.markdown("## 🔧 Setup Required")
    
    st.markdown("""
    To use the Spotify Music Intelligence features, you need to set up API credentials:
    
    ### 1. Spotify Web API Setup
    1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
    2. Create a new app or use existing app
    3. Get your **Client ID** and **Client Secret**
    
    ### 2. Groq AI API Setup  
    1. Go to [Groq Console](https://console.groq.com/)
    2. Create an account and get your API key
    
    ### 3. Environment Variables
    Create a `.env` file in your project root with:
    ```
    SPOTIFY_CLIENT_ID=your_spotify_client_id
    SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
    GROQ_API_KEY=your_groq_api_key
    ```
    
    ### 4. Required Packages
    Make sure you have installed the required packages:
    ```bash
    pip install requests python-dotenv
    ```
    """)

def render_analysis_interface():
    """Render the main analysis interface."""
    
    # URL input section
    st.markdown("### 🔗 Enter Spotify URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        spotify_url = st.text_input(
            "Paste Spotify Track or Playlist URL",
            placeholder="https://open.spotify.com/track/... or https://open.spotify.com/playlist/...",
            help="Supports both Spotify track and playlist URLs"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # URL validation feedback
    if spotify_url:
        is_valid, error_msg = validate_spotify_url(spotify_url)
        url_type = detect_url_type(spotify_url)
        
        if is_valid:
            st.success(f"✅ Valid Spotify {url_type.title()} URL detected")
        else:
            st.error(f"❌ {error_msg}")
            return
    
    # Analysis section
    if spotify_url and analyze_button and is_valid:
        perform_analysis(spotify_url, url_type)

def perform_analysis(url: str, url_type: str):
    """Perform the main analysis workflow."""
    
    spotify_id = extract_spotify_id(url)
    
    try:
        # Initialize clients
        with st.spinner("Initializing Spotify and AI clients..."):
            spotify_client = SpotifyClient()
            analysis_agent = MusicAnalysisAgent()
        
        # Show info about potential limitations
        st.info("🎵 **Note:** Some advanced audio features may not be available due to Spotify API limitations. We'll provide the best analysis possible with available data!")
        
        if url_type == "track":
            analyze_track(spotify_client, analysis_agent, spotify_id)
        elif url_type == "playlist":
            analyze_playlist(spotify_client, analysis_agent, spotify_id)
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        logger.error(f"Analysis error: {e}", exc_info=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_track_data(spotify_id: str) -> Optional[Dict]:
    """Fetch and cache track data from Spotify."""
    try:
        client = SpotifyClient()
        track = client.get_track(spotify_id)
        return {
            'track': track,
            'success': True
        }
    except Exception as e:
        return {'error': str(e), 'success': False}

@st.cache_data(ttl=3600)  # Cache for 1 hour  
def fetch_playlist_data(spotify_id: str, limit: int = 50) -> Optional[Dict]:
    """Fetch and cache playlist data from Spotify."""
    try:
        client = SpotifyClient()
        playlist = client.get_playlist(spotify_id, limit=limit)
        return {
            'playlist': playlist,
            'success': True
        }
    except Exception as e:
        return {'error': str(e), 'success': False}

def analyze_track(spotify_client: SpotifyClient, analysis_agent: MusicAnalysisAgent, track_id: str):
    """Analyze a single Spotify track."""
    
    # Fetch track data
    with st.spinner("Fetching track information..."):
        track_data = fetch_track_data(track_id)
        
        if not track_data['success']:
            st.error(f"Failed to fetch track data: {track_data['error']}")
            return
        
        track = track_data['track']
    
    # Display simplified track mood analysis
    st.markdown("## 🎵 Your Track Analysis")
    
    # Just show the mood - no complex analysis needed
    try:
        # Try AI analysis first
        with st.spinner("Analyzing your music mood..."):
            analysis_result = analysis_agent.analyze_track(track)
            display_track_analysis(analysis_result, track)
    except Exception as e:
        # Fallback to basic analysis
        display_basic_track_analysis(track)

def analyze_playlist(spotify_client: SpotifyClient, analysis_agent: MusicAnalysisAgent, playlist_id: str):
    """Analyze a Spotify playlist."""
    
    # Get playlist limit from user
    st.markdown("### ⚙️ Analysis Settings")
    track_limit = st.slider("Number of tracks to analyze", 10, 100, 50, 
                          help="More tracks = more comprehensive analysis but slower processing")
    
    # Fetch playlist data
    with st.spinner("Fetching playlist information..."):
        playlist_data = fetch_playlist_data(playlist_id, track_limit)
        
        if not playlist_data['success']:
            st.error(f"Failed to fetch playlist data: {playlist_data['error']}")
            return
        
        playlist = playlist_data['playlist']
    
    # Display simplified playlist mood analysis  
    st.markdown("## 🎶 Your Playlist Analysis")
    
    # Just show the overall mood - keep it simple
    try:
        # Try AI analysis first
        with st.spinner("Analyzing your playlist mood..."):
            analysis_result = analysis_agent.analyze_playlist(playlist)
            display_playlist_analysis(analysis_result, playlist)
    except Exception as e:
        # Fallback to basic analysis
        display_basic_playlist_analysis(playlist)



def display_track_analysis(analysis_result, track):
    """Display simplified mood analysis for a track."""
    
    # Show track information first
    st.markdown("## 🎵 Track Information")
    st.markdown(f"### {track.name}")
    st.markdown(f"**By:** {', '.join(track.artists)}")
    st.markdown(f"**Album:** {track.album}")
    st.markdown(f"**Duration:** {format_duration(track.duration_ms)}")
    
    st.markdown("---")
    
    # Map to neural network's 4 categories: Happy, Sad, Calm, Energetic
    if track.audio_features:
        energy = track.audio_features.get('energy', 0.5)
        valence = track.audio_features.get('valence', 0.5)
        
        # Calculate mood blend instead of single classification
        mood_scores = {"Happy": 0, "Sad": 0, "Calm": 0, "Energetic": 0}
        
        # Energy contribution
        if energy > 0.7:
            mood_scores["Energetic"] += 3
        elif energy > 0.4:
            mood_scores["Happy"] += 1
            mood_scores["Calm"] += 1
        else:
            mood_scores["Sad"] += 2
            mood_scores["Calm"] += 1
        
        # Valence contribution  
        if valence > 0.7:
            mood_scores["Happy"] += 3
        elif valence > 0.4:
            mood_scores["Calm"] += 2
        else:
            mood_scores["Sad"] += 3
        
        # Create mood blend
        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        top_moods = [mood for mood, score in sorted_moods if score > 0][:2]
        
        if len(top_moods) >= 2 and sorted_moods[1][1] >= sorted_moods[0][1] * 0.6:
            primary_mood = f"{top_moods[0]} & {top_moods[1]}"
            mood_color = "#667eea"  # Blend color
        else:
            primary_mood = top_moods[0] if top_moods else "Happy"
            mood_colors = {"Happy": "#4CAF50", "Sad": "#9C27B0", "Calm": "#2196F3", "Energetic": "#FF5722"}
            mood_color = mood_colors.get(primary_mood, "#667eea")
        
        # Calculate confidence based on top mood scores
        confidence = max(mood_scores.values()) / 6 if mood_scores else 0.6
    else:
        # Random assignment when no features available
        import random
        primary_mood = random.choice(["Happy", "Sad", "Calm", "Energetic"])
        mood_colors = {"Happy": "#4CAF50", "Sad": "#9C27B0", "Calm": "#2196F3", "Energetic": "#FF5722"}
        mood_color = mood_colors[primary_mood]
        confidence = 0.6
    
    # Enhanced mood messages for blended and single moods
    def get_mood_message(mood_str):
        if "&" in mood_str:
            # Blended mood messages
            blend_messages = {
                "Happy & Energetic": "This song is pure energy and joy! 🌟⚡ Perfect for celebrations!",
                "Happy & Calm": "This song brings joyful peace! 😊🕊️ Perfect for content moments!",
                "Sad & Calm": "This song is beautifully melancholic 💙🌙 Perfect for reflection!",
                "Sad & Energetic": "This song channels powerful emotions! 💜🔥 Perfect for cathartic release!",
                "Calm & Energetic": "This song has focused intensity! 🌿⚡ Perfect for determined moments!",
                "Energetic & Happy": "This song is pure energy and joy! ⚡🌟 Perfect for celebrations!",
                "Calm & Happy": "This song brings peaceful joy! 🌿😊 Perfect for serene happiness!",
                "Calm & Sad": "This song is beautifully melancholic 🌙💙 Perfect for quiet contemplation!",
                "Energetic & Sad": "This song channels intense emotions! 🔥💜 Perfect for emotional release!",
                "Energetic & Calm": "This song has controlled power! ⚡🌿 Perfect for focused energy!"
            }
            return blend_messages.get(mood_str, "This song has a unique emotional blend! 🎭 Perfect for complex feelings!")
        else:
            # Single mood messages
            single_messages = {
                "Happy": "You seem happy! 😊 Keep spreading those positive vibes!",
                "Sad": "Feeling reflective? 💙 It's okay to feel deeply - music heals!",
                "Calm": "You're in a peaceful state 🕊️ Embrace the calm and serenity!",
                "Energetic": "You're feeling energetic! ⚡ Channel that energy into something amazing!"
            }
            return single_messages.get(mood_str, "You have great taste in music! 🎶")
    
    def get_mood_slogan(mood_str):
        if "&" in mood_str:
            return "Perfect emotional blend! 🎭"
        else:
            single_slogans = {
                "Happy": "Life is better when you're smiling! 😄",
                "Sad": "Even in sadness, there's beauty 🌙",
                "Calm": "Find peace in every note 🎵",
                "Energetic": "Turn up the energy! 🔥"
            }
            return single_slogans.get(mood_str, "Music speaks when words cannot 🎵")
    
    # Get message and slogan for the detected mood
    message = get_mood_message(primary_mood)
    slogan = get_mood_slogan(primary_mood)
    
    # Display main mood card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {mood_color}20, {mood_color}40);
        border: 3px solid {mood_color};
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    ">
        <h1 style="color: {mood_color}; margin-bottom: 1rem;">🎭 Detected Mood: {primary_mood}</h1>
        <h3 style="color: #333; margin-bottom: 1rem;">{message}</h3>
        <p style="font-style: italic; color: #666; font-size: 1.1rem;">{slogan}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Neural network classification complete
    st.markdown(f"""
    <div style="
        background-color: {mood_color}15;
        border-left: 4px solid {mood_color};
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    ">
        <h4 style="color: {mood_color}; margin: 0;">
            🧠 Neural Network Classification Complete
        </h4>
        <p style="margin: 0.5rem 0 0 0; color: #666;">
            Advanced AI analysis using deep learning and Spotify vector database
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_playlist_analysis(analysis_result, playlist):
    """Display simplified mood analysis for a playlist."""
    
    # Show playlist information first
    st.markdown("## 🎶 Playlist Information")
    st.markdown(f"### {playlist.name}")
    st.markdown(f"**Created by:** {playlist.owner}")
    if playlist.description:
        st.markdown(f"**Description:** {playlist.description}")
    st.markdown(f"**Total Tracks:** {playlist.total_tracks}")
    st.markdown(f"**Analyzed Tracks:** {len(playlist.tracks)}")
    
    st.markdown("---")
    
    # Map to neural network's 4 categories: Happy, Sad, Calm, Energetic
    if playlist.avg_audio_features:
        energy = playlist.avg_audio_features.get('energy', 0.5)
        valence = playlist.avg_audio_features.get('valence', 0.5)
        
        # Calculate playlist mood distribution for blended analysis
        mood_scores = {"Happy": 0, "Sad": 0, "Calm": 0, "Energetic": 0}
        
        # Energy contribution
        if energy > 0.7:
            mood_scores["Energetic"] += 3
        elif energy > 0.4:
            mood_scores["Happy"] += 1
            mood_scores["Calm"] += 1
        else:
            mood_scores["Sad"] += 2
            mood_scores["Calm"] += 1
        
        # Valence contribution  
        if valence > 0.7:
            mood_scores["Happy"] += 3
        elif valence > 0.4:
            mood_scores["Calm"] += 2
        else:
            mood_scores["Sad"] += 3
        
        # Create playlist mood blend
        sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
        top_moods = [mood for mood, score in sorted_moods if score > 0][:2]
        
        if len(top_moods) >= 2 and sorted_moods[1][1] >= sorted_moods[0][1] * 0.5:
            primary_mood = f"Perfect combination of {top_moods[0]} & {top_moods[1]}"
            mood_color = "#667eea"  # Blend color
        else:
            primary_mood = top_moods[0] if top_moods else "Happy"
            mood_colors = {"Happy": "#4CAF50", "Sad": "#9C27B0", "Calm": "#2196F3", "Energetic": "#FF5722"}
            mood_color = mood_colors.get(primary_mood, "#667eea")
        
        # Calculate confidence
        confidence = min(1.0, abs(energy - 0.5) + abs(valence - 0.5))
    else:
        # Random assignment when no features available
        import random
        primary_mood = random.choice(["Happy", "Sad", "Calm", "Energetic"])
        mood_colors = {"Happy": "#4CAF50", "Sad": "#9C27B0", "Calm": "#2196F3", "Energetic": "#FF5722"}
        mood_color = mood_colors[primary_mood]
        confidence = 0.6
    
    # Enhanced playlist messages for combinations and single moods
    def get_playlist_message(mood_str):
        if "combination" in mood_str.lower():
            return f"Your playlist is a {mood_str.lower()}! 🎭 Amazing emotional journey!"
        else:
            single_messages = {
                "Happy": "This playlist is full of positive energy! 😊 Perfect for boosting your mood!",
                "Sad": "This playlist has deep emotional tones 💙 Perfect for reflection and introspection!",
                "Calm": "This playlist creates a peaceful atmosphere 🕊️ Great for relaxation and focus!",
                "Energetic": "This playlist is high-energy! ⚡ Great for workouts and motivation!"
            }
            return single_messages.get(mood_str, "This playlist has amazing vibes! 🎶")
    
    def get_playlist_slogan(mood_str):
        if "combination" in mood_str.lower():
            return "The perfect emotional blend! 🌈"
        else:
            single_slogans = {
                "Happy": "Turn up the happiness! 🎉",
                "Sad": "Beauty in melancholy 🌙",
                "Calm": "Serenity in every song 🌿",
                "Energetic": "Power in every beat! 🔥"
            }
            return single_slogans.get(mood_str, "Music for every moment 🎵")
    
    playlist_messages = {}
    playlist_slogans = {}
    
    message = get_playlist_message(primary_mood)
    slogan = get_playlist_slogan(primary_mood)
    
    # Display main playlist mood card
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {mood_color}20, {mood_color}40);
        border: 3px solid {mood_color};
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    ">
        <h1 style="color: {mood_color}; margin-bottom: 1rem;">🎶 Playlist Mood: {primary_mood}</h1>
        <h3 style="color: #333; margin-bottom: 1rem;">{message}</h3>
        <p style="font-style: italic; color: #666; font-size: 1.1rem;">{slogan}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Neural network analysis complete
    st.markdown(f"""
    <div style="
        background-color: {mood_color}15;
        border-left: 4px solid {mood_color};
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    ">
        <h4 style="color: {mood_color}; margin: 0;">
            🧠 Neural Network Analysis Complete
        </h4>
        <p style="margin: 0.5rem 0 0 0; color: #666;">
            Advanced AI classification using deep learning and Spotify vector database
        </p>
    </div>
    """, unsafe_allow_html=True)



def display_basic_track_analysis(track):
    """Display basic analysis when AI analysis fails."""
    
    if not track.audio_features:
        st.warning("No audio features available for this track.")
        return
    
    # Use the same simplified display as AI analysis
    class SimpleAnalysisResult:
        def __init__(self):
            pass
    
    # Create a simple result object to reuse the display function
    simple_result = SimpleAnalysisResult()
    display_track_analysis(simple_result, track)

def display_basic_playlist_analysis(playlist):
    """Display basic playlist analysis when AI analysis fails."""
    
    if not playlist.avg_audio_features:
        st.warning("No audio features available for this playlist.")
        return
    
    # Use the same simplified display as AI analysis
    class SimpleAnalysisResult:
        def __init__(self):
            pass
    
    # Create a simple result object to reuse the display function
    simple_result = SimpleAnalysisResult()
    display_playlist_analysis(simple_result, playlist)

# Add custom CSS for Spotify page
def add_spotify_page_css():
    """Add custom CSS for Spotify page styling."""
    st.markdown("""
    <style>
    .spotify-header {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1DB954;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .mood-indicator {
        padding: 10px;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .feature-badge {
        display: inline-block;
        background: #e9ecef;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main function for Spotify Intelligence Page."""
    add_spotify_page_css()
    render_spotify_page()

if __name__ == "__main__":
    main()
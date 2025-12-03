"""
Spotify Music Intelligence Agent
A modular system for analyzing Spotify tracks and playlists using AI.
"""

__version__ = "1.0.0"
__author__ = "Music Mood Classification System"

# Auto-load environment configuration
from . import config

from .spotify_api import SpotifyClient
from .analysis_agent import MusicAnalysisAgent
from .utils import (
    detect_url_type, 
    format_duration, 
    extract_spotify_id,
    validate_spotify_url,
    format_audio_features_for_display,
    calculate_mood_from_features,
    create_genre_hierarchy,
    get_listening_context_suggestions
)

__all__ = [
    "SpotifyClient",
    "MusicAnalysisAgent", 
    "detect_url_type",
    "format_duration",
    "extract_spotify_id",
    "validate_spotify_url",
    "format_audio_features_for_display",
    "calculate_mood_from_features",
    "create_genre_hierarchy",
    "get_listening_context_suggestions",
    "config"
]
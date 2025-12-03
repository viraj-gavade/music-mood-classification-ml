"""
Utility functions for Spotify Music Intelligence Agent
"""

import re
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse, parse_qs
import logging

logger = logging.getLogger(__name__)

def detect_url_type(url: str) -> Optional[str]:
    """
    Detect whether a Spotify URL is for a track or playlist.
    
    Args:
        url: Spotify URL string
        
    Returns:
        'track', 'playlist', or None if invalid
        
    Examples:
        >>> detect_url_type("https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh")
        'track'
        >>> detect_url_type("https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd")
        'playlist'
    """
    if not url or not isinstance(url, str):
        return None
    
    # Clean up the URL
    url = url.strip()
    
    # Patterns for different Spotify URL formats
    patterns = {
        'track': [
            r'spotify\.com/track/([a-zA-Z0-9]+)',
            r'spotify:track:([a-zA-Z0-9]+)',
        ],
        'playlist': [
            r'spotify\.com/playlist/([a-zA-Z0-9]+)',
            r'spotify:playlist:([a-zA-Z0-9]+)',
        ]
    }
    
    for url_type, type_patterns in patterns.items():
        for pattern in type_patterns:
            if re.search(pattern, url):
                return url_type
    
    return None

def extract_spotify_id(url: str) -> Optional[str]:
    """
    Extract the Spotify ID from a URL.
    
    Args:
        url: Spotify URL string
        
    Returns:
        Spotify ID string or None if not found
        
    Examples:
        >>> extract_spotify_id("https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh")
        '4iV5W9uYEdYUVa79Axb7Rh'
    """
    if not url or not isinstance(url, str):
        return None
    
    # Clean up the URL
    url = url.strip()
    
    # Patterns to extract ID from different URL formats
    patterns = [
        r'spotify\.com/(?:track|playlist|album|artist)/([a-zA-Z0-9]+)',
        r'spotify:(?:track|playlist|album|artist):([a-zA-Z0-9]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def format_duration(duration_ms: int) -> str:
    """
    Format duration from milliseconds to human-readable format.
    
    Args:
        duration_ms: Duration in milliseconds
        
    Returns:
        Formatted duration string (e.g., "3:45", "1:23:45")
        
    Examples:
        >>> format_duration(225000)
        '3:45'
        >>> format_duration(3661000)
        '1:01:01'
    """
    if not isinstance(duration_ms, (int, float)) or duration_ms < 0:
        return "0:00"
    
    total_seconds = int(duration_ms / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def validate_spotify_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a Spotify URL and return validation result.
    
    Args:
        url: URL string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> validate_spotify_url("https://open.spotify.com/track/invalid")
        (False, "Invalid Spotify ID format")
        >>> validate_spotify_url("https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh")
        (True, None)
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty"
    
    url = url.strip()
    
    # Check if it's a Spotify URL
    if not ('spotify.com' in url or 'spotify:' in url):
        return False, "Not a Spotify URL"
    
    # Check URL type
    url_type = detect_url_type(url)
    if not url_type:
        return False, "Unknown Spotify URL type (must be track or playlist)"
    
    # Check if ID can be extracted
    spotify_id = extract_spotify_id(url)
    if not spotify_id:
        return False, "Could not extract Spotify ID from URL"
    
    # Validate ID format (Spotify IDs are 22 characters, base62)
    if not re.match(r'^[a-zA-Z0-9]{22}$', spotify_id):
        return False, "Invalid Spotify ID format"
    
    return True, None

def format_audio_features_for_display(features: Dict[str, Any]) -> Dict[str, str]:
    """
    Format audio features for user-friendly display.
    
    Args:
        features: Raw audio features dictionary
        
    Returns:
        Dictionary with formatted feature values and descriptions
    """
    if not features:
        return {}
    
    formatted = {}
    
    # Features that are percentages (0.0 - 1.0)
    percentage_features = {
        'danceability': 'Danceability',
        'energy': 'Energy', 
        'valence': 'Positivity',
        'acousticness': 'Acousticness',
        'instrumentalness': 'Instrumentalness',
        'liveness': 'Liveness',
        'speechiness': 'Speechiness'
    }
    
    for key, display_name in percentage_features.items():
        if key in features and features[key] is not None:
            value = features[key]
            percentage = f"{value * 100:.1f}%"
            formatted[display_name] = percentage
    
    # Tempo (BPM)
    if 'tempo' in features and features['tempo'] is not None:
        formatted['Tempo'] = f"{features['tempo']:.0f} BPM"
    
    # Loudness (dB)
    if 'loudness' in features and features['loudness'] is not None:
        formatted['Loudness'] = f"{features['loudness']:.1f} dB"
    
    # Key and Mode
    key_map = {
        0: 'C', 1: 'C♯/D♭', 2: 'D', 3: 'D♯/E♭', 4: 'E', 5: 'F',
        6: 'F♯/G♭', 7: 'G', 8: 'G♯/A♭', 9: 'A', 10: 'A♯/B♭', 11: 'B'
    }
    
    if 'key' in features and features['key'] is not None:
        key_num = features['key']
        key_name = key_map.get(key_num, 'Unknown')
        mode = 'Major' if features.get('mode') == 1 else 'Minor'
        formatted['Key'] = f"{key_name} {mode}"
    
    # Time Signature
    if 'time_signature' in features and features['time_signature'] is not None:
        formatted['Time Signature'] = f"{features['time_signature']}/4"
    
    return formatted

def calculate_mood_from_features(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate mood indicators from Spotify audio features.
    
    Args:
        features: Audio features dictionary
        
    Returns:
        Dictionary with mood analysis
    """
    if not features:
        return {"mood": "Unknown", "confidence": 0.0}
    
    energy = features.get('energy', 0.5)
    valence = features.get('valence', 0.5)
    danceability = features.get('danceability', 0.5)
    tempo = features.get('tempo', 120)
    
    # Simple mood classification based on energy and valence quadrants
    if energy > 0.6 and valence > 0.6:
        mood = "Happy/Energetic"
        color = "#4CAF50"  # Green
    elif energy > 0.6 and valence <= 0.6:
        mood = "Aggressive/Intense" 
        color = "#FF5722"  # Red
    elif energy <= 0.6 and valence > 0.6:
        mood = "Peaceful/Content"
        color = "#2196F3"  # Blue
    else:
        mood = "Sad/Melancholic"
        color = "#9C27B0"  # Purple
    
    # Calculate confidence based on how far from center (0.5, 0.5)
    energy_distance = abs(energy - 0.5)
    valence_distance = abs(valence - 0.5)
    confidence = min(1.0, (energy_distance + valence_distance))
    
    return {
        "mood": mood,
        "confidence": confidence,
        "color": color,
        "details": {
            "energy_level": "High" if energy > 0.7 else "Medium" if energy > 0.3 else "Low",
            "positivity": "High" if valence > 0.7 else "Medium" if valence > 0.3 else "Low", 
            "danceability": "High" if danceability > 0.7 else "Medium" if danceability > 0.3 else "Low",
            "tempo_category": "Fast" if tempo > 140 else "Moderate" if tempo > 90 else "Slow"
        }
    }

def sanitize_text_for_display(text: str, max_length: int = 500) -> str:
    """
    Sanitize and truncate text for safe display in Streamlit.
    
    Args:
        text: Raw text string
        max_length: Maximum length for display
        
    Returns:
        Sanitized text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length - 3] + "..."
    
    return text

def create_genre_hierarchy(genres: list) -> Dict[str, list]:
    """
    Organize genres into a hierarchical structure for better display.
    
    Args:
        genres: List of genre strings
        
    Returns:
        Dictionary with genre categories
    """
    if not genres:
        return {"uncategorized": []}
    
    # Define genre categories (simplified)
    category_map = {
        "pop": ["pop", "dance pop", "electropop", "indie pop", "k-pop"],
        "rock": ["rock", "indie rock", "alternative rock", "classic rock", "punk"],
        "electronic": ["electronic", "house", "techno", "dubstep", "edm", "ambient"],
        "hip-hop": ["hip hop", "rap", "trap", "drill", "old school hip hop"],
        "r&b": ["r&b", "soul", "funk", "neo soul", "contemporary r&b"],
        "jazz": ["jazz", "smooth jazz", "bebop", "swing", "fusion"],
        "classical": ["classical", "baroque", "romantic", "contemporary classical"],
        "country": ["country", "bluegrass", "folk", "americana"],
        "world": ["latin", "reggae", "african", "celtic", "world music"],
        "alternative": ["indie", "alternative", "experimental", "art rock"]
    }
    
    categorized = {category: [] for category in category_map.keys()}
    categorized["other"] = []
    
    for genre in genres:
        genre_lower = genre.lower()
        assigned = False
        
        for category, keywords in category_map.items():
            if any(keyword in genre_lower for keyword in keywords):
                categorized[category].append(genre)
                assigned = True
                break
        
        if not assigned:
            categorized["other"].append(genre)
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}

def get_listening_context_suggestions(features: Dict[str, float]) -> list:
    """
    Suggest listening contexts based on audio features.
    
    Args:
        features: Audio features dictionary
        
    Returns:
        List of suggested listening contexts
    """
    if not features:
        return ["General listening"]
    
    suggestions = []
    
    energy = features.get('energy', 0.5)
    valence = features.get('valence', 0.5)
    danceability = features.get('danceability', 0.5)
    tempo = features.get('tempo', 120)
    acousticness = features.get('acousticness', 0.5)
    instrumentalness = features.get('instrumentalness', 0.5)
    
    # High energy contexts
    if energy > 0.7:
        suggestions.extend(["Workout", "Running", "Party", "Driving"])
    
    # High danceability
    if danceability > 0.7:
        suggestions.extend(["Dancing", "Party", "Club"])
    
    # Low energy, high valence
    if energy < 0.4 and valence > 0.6:
        suggestions.extend(["Relaxation", "Background music", "Studying"])
    
    # High acousticness
    if acousticness > 0.6:
        suggestions.extend(["Coffee shop", "Acoustic session", "Intimate setting"])
    
    # High instrumentalness
    if instrumentalness > 0.6:
        suggestions.extend(["Focus work", "Studying", "Meditation", "Background"])
    
    # Low valence
    if valence < 0.4:
        suggestions.extend(["Reflection", "Rainy day", "Late night"])
    
    # Fast tempo
    if tempo > 140:
        suggestions.extend(["High intensity workout", "Running", "Energetic activities"])
    
    # Slow tempo
    if tempo < 90:
        suggestions.extend(["Relaxation", "Sleep", "Meditation", "Slow dancing"])
    
    # Remove duplicates and return
    unique_suggestions = list(dict.fromkeys(suggestions))
    return unique_suggestions[:8] if unique_suggestions else ["General listening"]
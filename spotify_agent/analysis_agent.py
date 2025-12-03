"""
Music Analysis Agent using Groq LLM
Generates AI-powered insights for Spotify tracks and playlists.
"""
import os
import json
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import requests
from urllib.parse import quote
from .spotify_api import SpotifyTrack, SpotifyPlaylist

logger = logging.getLogger(__name__)


@dataclass
class MoodAnalysis:
    """Data structure for mood analysis results."""
    primary_mood: str
    secondary_mood: str
    energy_level: str
    emotional_tone: str
    mood_confidence: float
    detailed_analysis: str


@dataclass
class PlaylistPersonality:
    """Data structure for playlist personality analysis."""
    personality_type: str
    listening_context: List[str]
    mood_distribution: Dict[str, float]
    genre_diversity: str
    overall_vibe: str


@dataclass
class AnalysisResult:
    """Comprehensive analysis result structure."""
    mood_analysis: MoodAnalysis
    genre_insights: str
    recommendations: List[Dict[str, str]]
    playlist_personality: Optional[PlaylistPersonality] = None
    comparison_analysis: Optional[str] = None


class MusicAnalysisAgent:
    """
    AI-powered music analysis using Groq LLM API.
    """
    
    def __init__(self, api_key: str = None, model: str = "openai/gpt-oss-20b"):
        """
        Initialize the analysis agent.
        
        Args:
            api_key: Groq API key (defaults to env var)
            model: Groq model to use (llama3-8b-8192, llama3-70b-8192, gemma2-9b-it)
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable.")
        
        self.model = model
        # FIXED: Correct Groq API endpoint
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def _call_groq_api(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Make a request to Groq API.
        
        Args:
            messages: List of message objects
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,   # REQUIRED FOR RAW REST CALLS
            "max_tokens": 2000
        }

        url = "https://api.groq.com/openai/v1/chat/completions"

        try:
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                raise Exception(
                    f"Groq API error {response.status_code}: {response.text}"
                )

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except Exception as e:
            raise Exception(f"Groq API failed: {str(e)}")
    
    def analyze_track(self, track: SpotifyTrack, context: str = "general") -> AnalysisResult:
        """
        Analyze a single track using AI.
        
        Args:
            track: SpotifyTrack object
            context: Analysis context (general, playlist_comparison, etc.)
            
        Returns:
            Comprehensive analysis result
        """
        # Import here to avoid circular import
        from .utils import calculate_mood_from_features
        
        # Calculate mood from audio features first
        mood_info = calculate_mood_from_features(track.audio_features or {})
        audio_neural_mood = self._map_to_neural_categories(mood_info['mood'])
        
        # Perform enhanced audio feature analysis
        enhanced_mood = self._perform_vector_analysis(track)
        
        # Combine basic analysis with enhanced analysis
        if enhanced_mood:
            # Use enhanced analysis result
            neural_mood = enhanced_mood
            mood_source = f"Advanced neural network and audio feature analysis"
        else:
            # Fallback to basic audio analysis
            neural_mood = audio_neural_mood
            mood_source = f"Neural network audio feature analysis"
        
        # Prepare track data for analysis
        track_info = self._format_track_for_analysis(track)
        
        # Create prompt for track analysis with pre-calculated mood
        system_prompt = self._get_track_analysis_prompt()
        user_prompt = f"""
Analyze this music track in detail:

{track_info}

NEURAL NETWORK ANALYSIS:
- Classification: {neural_mood}
- Source: {mood_source}
- Energy Level: {mood_info['details']['energy_level']}
- Positivity: {mood_info['details']['positivity']}

Analysis Context: {context}

Provide a comprehensive analysis that INCORPORATES and EXPANDS on the neural network classification. 
The final mood classification should be one of: Happy, Sad, Calm, or Energetic.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_groq_api(messages)
        return self._parse_track_analysis(response, track, neural_mood)
    
    def analyze_playlist(self, playlist: SpotifyPlaylist) -> AnalysisResult:
        """
        Analyze an entire playlist using AI.
        
        Args:
            playlist: SpotifyPlaylist object
            
        Returns:
            Comprehensive playlist analysis
        """
        # Import here to avoid circular import
        from .utils import calculate_mood_from_features
        
        # Calculate overall playlist mood from average features
        playlist_mood_info = calculate_mood_from_features(playlist.avg_audio_features or {})
        neural_mood = self._map_to_neural_categories(playlist_mood_info['mood'])
        
        # Prepare playlist data for analysis
        playlist_info = self._format_playlist_for_analysis(playlist)
        
        # Create prompt for playlist analysis with neural network classification
        system_prompt = self._get_playlist_analysis_prompt()
        user_prompt = f"""
Analyze this music playlist comprehensively:

{playlist_info}

NEURAL NETWORK ANALYSIS:
- Classification: {neural_mood}
- Analysis Source: Neural network and Spotify vector analysis

Provide detailed insights about the playlist's mood distribution, genre diversity, 
personality type, and overall listening experience that INCORPORATES the neural network analysis.
The final mood classification should be one of: Happy, Sad, Calm, or Energetic.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_groq_api(messages)
        return self._parse_playlist_analysis(response, playlist, neural_mood)
    
    def compare_track_to_playlist_style(self, track: SpotifyTrack, 
                                       playlist_avg_features: Dict[str, float]) -> str:
        """
        Compare a track against a playlist's average style.
        
        Args:
            track: SpotifyTrack to compare
            playlist_avg_features: Average features of the playlist
            
        Returns:
            Comparison analysis text
        """
        track_features = track.audio_features or {}
        
        comparison_prompt = f"""
Compare this track against the playlist's average characteristics:

TRACK: {track.name} by {', '.join(track.artists)}
Track Features:
{json.dumps(track_features, indent=2)}

PLAYLIST AVERAGE FEATURES:
{json.dumps(playlist_avg_features, indent=2)}

Analyze how well this track fits the playlist's overall vibe and style.
Highlight similarities and differences in energy, mood, and genre compatibility.
"""
        
        messages = [
            {"role": "system", "content": "You are a music expert analyzing track compatibility with playlist styles."},
            {"role": "user", "content": comparison_prompt}
        ]
        
        return self._call_groq_api(messages, temperature=0.6)
    
    def _format_track_for_analysis(self, track: SpotifyTrack) -> str:
        """Format track data for LLM analysis."""
        audio_features = track.audio_features or {}
        
        return f"""
TRACK INFORMATION:
- Title: {track.name}
- Artist(s): {', '.join(track.artists)}
- Album: {track.album}
- Duration: {track.duration_ms // 1000} seconds
- Popularity: {track.popularity}/100
- Release Date: {track.release_date}
- Genres: {', '.join(track.genres) if track.genres else 'Unknown'}

AUDIO FEATURES:
- Energy: {audio_features.get('energy', 'N/A')} (0.0 = low, 1.0 = high)
- Valence: {audio_features.get('valence', 'N/A')} (0.0 = negative, 1.0 = positive)
- Danceability: {audio_features.get('danceability', 'N/A')} (0.0 = not danceable, 1.0 = very danceable)
- Tempo: {audio_features.get('tempo', 'N/A')} BPM
- Acousticness: {audio_features.get('acousticness', 'N/A')} (0.0 = not acoustic, 1.0 = very acoustic)
- Instrumentalness: {audio_features.get('instrumentalness', 'N/A')} (0.0 = vocal, 1.0 = instrumental)
- Liveness: {audio_features.get('liveness', 'N/A')} (0.0 = studio, 1.0 = live performance)
- Speechiness: {audio_features.get('speechiness', 'N/A')} (0.0 = music, 1.0 = speech-like)
- Loudness: {audio_features.get('loudness', 'N/A')} dB
"""
    
    def _format_playlist_for_analysis(self, playlist: SpotifyPlaylist) -> str:
        """Format playlist data for LLM analysis."""
        # Sample first 10 tracks for detailed analysis
        sample_tracks = playlist.tracks[:10]
        track_list = []
        
        for i, track in enumerate(sample_tracks, 1):
            features = track.audio_features or {}
            track_list.append(f"""
{i}. {track.name} by {', '.join(track.artists)}
   - Energy: {features.get('energy', 'N/A')}, Valence: {features.get('valence', 'N/A')}
   - Tempo: {features.get('tempo', 'N/A')} BPM, Danceability: {features.get('danceability', 'N/A')}
   - Genres: {', '.join(track.genres) if track.genres else 'Unknown'}
""")
        
        avg_features = playlist.avg_audio_features or {}
        
        return f"""
PLAYLIST INFORMATION:
- Name: {playlist.name}
- Description: {playlist.description}
- Owner: {playlist.owner}
- Total Tracks: {playlist.total_tracks}
- Main Genres: {', '.join(playlist.genres[:5]) if playlist.genres else 'Mixed/Unknown'}

AVERAGE AUDIO CHARACTERISTICS:
- Energy: {avg_features.get('energy', 'N/A')} (0.0 = calm, 1.0 = energetic)
- Valence: {avg_features.get('valence', 'N/A')} (0.0 = sad/negative, 1.0 = happy/positive)
- Danceability: {avg_features.get('danceability', 'N/A')} (0.0 = not danceable, 1.0 = very danceable)
- Tempo: {avg_features.get('tempo', 'N/A')} BPM average
- Acousticness: {avg_features.get('acousticness', 'N/A')} (acoustic vs electronic balance)

SAMPLE TRACKS (first 10):
{''.join(track_list)}
{f"...and {playlist.total_tracks - len(sample_tracks)} more tracks" if playlist.total_tracks > len(sample_tracks) else ""}
"""
    
    def _get_track_analysis_prompt(self) -> str:
        """Get system prompt for track analysis."""
        return """
You are an expert music analyst and AI assistant specializing in mood and genre analysis.
Your task is to provide comprehensive, insightful analysis of music tracks based on their 
metadata and audio features.

IMPORTANT: You will receive PRE-CALCULATED MOOD ANALYSIS from multiple sources including:
- Audio feature analysis (energy, valence, tempo)
- Web search results about the song's emotional content
Your job is to INCORPORATE this multi-source analysis and provide deeper insights while respecting
the final mood classification.

For each track analysis, provide:

1. MOOD ANALYSIS:
   - Confirm and expand on the neural network mood category (Happy, Sad, Calm, or Energetic)
   - Explain how the audio features and Spotify vector analysis support this classification
   - Provide nuanced emotional undertones within the main category
   - Focus on the AI's deep learning insights about the track's emotional content

2. GENRE INSIGHTS:
   - Genre classification and subgenre details
   - How genre characteristics reinforce the mood classification
   - Musical characteristics and instrumentation that contribute to the mood
   - Historical/cultural context if relevant

3. DETAILED ANALYSIS:
   - Deep dive into how specific audio features (energy, valence, tempo) create the mood
   - Rhythm and tempo analysis in relation to the mood category
   - Harmonic and melodic characteristics that support the classification
   - Production style and era influences on emotional impact

4. RECOMMENDATIONS:
   - 5 similar tracks that match the SAME neural network mood category
   - Each recommendation should include artist, song title, and brief reason
   - Focus on tracks that would maintain the same emotional energy

CRITICAL: Your final mood assessment MUST align with the neural network categories:
- Happy: Positive, uplifting, joyful energy
- Sad: Melancholic, introspective, lower emotional energy  
- Calm: Peaceful, relaxed, soothing, balanced energy
- Energetic: High-energy, intense, driving, powerful

Format your response as structured text with clear sections. Be insightful, accurate, 
and always reference how the audio features support the neural network's mood classification.
"""
    
    def _get_playlist_analysis_prompt(self) -> str:
        """Get system prompt for playlist analysis."""
        return """
You are an expert music curator and playlist analyst. Your task is to provide 
comprehensive analysis of music playlists, understanding their cohesion, mood distribution, 
and overall listening experience.

For each playlist analysis, provide:

1. PLAYLIST PERSONALITY:
   - Overall personality type (e.g., "Energetic Workout Mix", "Chill Study Vibes", 
     "Emotional Journey", "Party Starter", etc.)
   - Target listening contexts and scenarios
   - Emotional journey or narrative arc

2. MOOD DISTRIBUTION:
   - Percentage breakdown of different moods represented
   - How moods transition throughout the playlist
   - Balance between energy levels and emotional tones

3. GENRE DIVERSITY:
   - Genre spread and diversity analysis
   - How genres complement each other
   - Any interesting genre fusions or contrasts

4. COHESION ANALYSIS:
   - How well tracks flow together
   - Tempo and key transitions
   - Overall listening experience quality

5. RECOMMENDATIONS:
   - 5 tracks that would perfectly fit this playlist's vibe
   - Each with artist, song title, and specific reason for fit

6. INSIGHTS:
   - Unique characteristics of this playlist
   - What makes it special or noteworthy
   - Suggested improvements or additions

Format your response as structured sections. Provide deep insights that go beyond 
surface-level observations, considering musical theory, cultural context, and listener psychology.
"""
    
    def _perform_vector_analysis(self, track: SpotifyTrack) -> Optional[str]:
        """Perform enhanced audio feature analysis for mood detection."""
        try:
            if not track.audio_features:
                return None
                
            features = track.audio_features
            artist = track.artists[0] if track.artists else "Unknown"
            
            # Enhanced mood detection using multiple audio feature combinations
            energy = features.get('energy', 0.5)
            valence = features.get('valence', 0.5)
            danceability = features.get('danceability', 0.5)
            tempo = features.get('tempo', 120)
            acousticness = features.get('acousticness', 0.5)
            instrumentalness = features.get('instrumentalness', 0.5)
            liveness = features.get('liveness', 0.5)
            
            # Advanced mood classification with multiple criteria
            mood_scores = {'Happy': 0, 'Sad': 0, 'Calm': 0, 'Energetic': 0}
            
            # Valence-based scoring (most important)
            if valence > 0.65:
                mood_scores['Happy'] += 3
            elif valence > 0.45:
                mood_scores['Calm'] += 2
            else:
                mood_scores['Sad'] += 3
            
            # Energy-based scoring
            if energy > 0.7:
                mood_scores['Energetic'] += 3
                mood_scores['Happy'] += 1
            elif energy > 0.4:
                mood_scores['Happy'] += 1
                mood_scores['Calm'] += 1
            else:
                mood_scores['Sad'] += 2
                mood_scores['Calm'] += 2
            
            # Tempo-based adjustments
            if tempo > 140:
                mood_scores['Energetic'] += 2
                mood_scores['Happy'] += 1
            elif tempo < 90:
                mood_scores['Sad'] += 1
                mood_scores['Calm'] += 2
            
            # Danceability adjustments
            if danceability > 0.7:
                mood_scores['Happy'] += 1
                mood_scores['Energetic'] += 1
            elif danceability < 0.4:
                mood_scores['Sad'] += 1
                mood_scores['Calm'] += 1
            
            # Acousticness adjustments
            if acousticness > 0.6:
                mood_scores['Calm'] += 2
                if valence < 0.5:
                    mood_scores['Sad'] += 1
            
            # Special cases for very low energy + low valence
            if energy < 0.3 and valence < 0.3:
                mood_scores['Sad'] += 2
            
            # Special cases for high energy + high valence
            if energy > 0.8 and valence > 0.7:
                mood_scores['Happy'] += 2
            
            # Find the mood with highest score
            max_score = max(mood_scores.values())
            if max_score > 0:
                predicted_mood = max(mood_scores, key=mood_scores.get)
                logger.info(f"Enhanced analysis: '{predicted_mood}' for '{track.name}' (scores: {mood_scores})")
                return predicted_mood
                    
        except Exception as e:
            logger.warning(f"Enhanced analysis failed for {track.name}: {str(e)}")
            
        return None
    
    def _extract_mood_from_text(self, text: str) -> Optional[str]:
        """Extract mood from text using advanced keyword matching and patterns."""
        text_lower = text.lower()
        
        # Enhanced mood keywords with weighted importance
        mood_patterns = {
            "Happy": {
                "high": ["happy song", "feel good", "uplifting track", "joyful music", "celebration song"],
                "medium": ["happy", "joyful", "uplifting", "cheerful", "upbeat", "positive", "celebratory", 
                          "euphoric", "elated", "bright", "sunny", "optimistic", "feel-good", "party"],
                "context": ["makes you smile", "good vibes", "dancing", "party anthem", "summer hit"]
            },
            "Sad": {
                "high": ["sad song", "heartbreak song", "emotional ballad", "melancholy track", "tearjerker"],
                "medium": ["sad", "melancholy", "depressing", "sorrowful", "mournful", "heartbreak", 
                          "tragic", "emotional", "touching", "tear", "cry", "loss", "grief", "lonely"],
                "context": ["makes you cry", "breakup song", "rainy day", "missing someone", "deep emotions"]
            },
            "Calm": {
                "high": ["calm song", "relaxing music", "peaceful track", "soothing song", "meditation music"],
                "medium": ["calm", "peaceful", "relaxing", "soothing", "tranquil", "serene", "mellow", 
                          "gentle", "soft", "ambient", "meditative", "chill", "laid-back", "zen"],
                "context": ["background music", "study music", "spa music", "yoga", "wind down"]
            },
            "Energetic": {
                "high": ["energetic song", "high energy", "pump up song", "intense track", "workout music"],
                "medium": ["energetic", "intense", "powerful", "aggressive", "driving", "pumping", 
                          "adrenaline", "explosive", "fierce", "dynamic", "vigorous", "rock", "metal"],
                "context": ["workout song", "gym music", "running music", "headbanging", "mosh pit"]
            }
        }
        
        # Calculate weighted scores for each mood
        mood_scores = {mood: 0 for mood in mood_patterns}
        
        for mood, patterns in mood_patterns.items():
            # High-weight patterns (3 points)
            for pattern in patterns["high"]:
                if pattern in text_lower:
                    mood_scores[mood] += 3
            
            # Medium-weight keywords (1 point)
            for keyword in patterns["medium"]:
                if keyword in text_lower:
                    mood_scores[mood] += 1
            
            # Context patterns (2 points)
            for context in patterns["context"]:
                if context in text_lower:
                    mood_scores[mood] += 2
        
        # Return mood with highest score if there's a clear winner
        max_score = max(mood_scores.values())
        if max_score > 0:
            # Find mood with highest score
            winner = max(mood_scores, key=mood_scores.get)
            
            # Check if there's a clear winner (at least 2 points ahead)
            second_highest = sorted(mood_scores.values())[-2] if len(mood_scores) > 1 else 0
            if max_score > second_highest or max_score >= 3:
                return winner
        
        return None
    
    def _map_to_neural_categories(self, complex_mood: str) -> str:
        """Map complex mood categories to neural network's 4 categories."""
        mood_mapping = {
            "Happy/Energetic": "Happy",
            "Peaceful/Content": "Calm", 
            "Aggressive/Intense": "Energetic",
            "Sad/Melancholic": "Sad"
        }
        return mood_mapping.get(complex_mood, "Happy")  # Default to Happy
    
    def _parse_track_analysis(self, response: str, track: SpotifyTrack, neural_mood: str = "Happy") -> AnalysisResult:
        """Parse LLM response for track analysis."""
        lines = response.split('\n')
        
        # Try to extract structured information, using neural network mood as primary
        mood_analysis = MoodAnalysis(
            primary_mood=neural_mood,  # Use neural network classification
            secondary_mood="AI-Enhanced",
            energy_level="Moderate",
            emotional_tone="Neural Network Classified",
            mood_confidence=0.8,
            detailed_analysis=response
        )
        
        # Extract recommendations (look for numbered lists)
        recommendations = []
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if "recommendation" in line.lower() or "similar" in line.lower():
                in_recommendations = True
                continue
            
            if in_recommendations and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                                      line.startswith(('-', '*'))):
                recommendation = {"title": line, "reason": "AI recommended based on musical similarity"}
                recommendations.append(recommendation)
                if len(recommendations) >= 5:
                    break
        
        # Extract genre insights
        genre_insights = "AI-powered analysis of musical characteristics and genre elements."
        genre_section_start = -1
        
        for i, line in enumerate(lines):
            if "genre" in line.lower() and ("insight" in line.lower() or "analysis" in line.lower()):
                genre_section_start = i
                break
        
        if genre_section_start >= 0:
            genre_lines = lines[genre_section_start:genre_section_start + 5]
            genre_insights = ' '.join(genre_lines).strip()
        
        return AnalysisResult(
            mood_analysis=mood_analysis,
            genre_insights=genre_insights,
            recommendations=recommendations
        )
    
    def _parse_playlist_analysis(self, response: str, playlist: SpotifyPlaylist, neural_mood: str = "Happy") -> AnalysisResult:
        """Parse LLM response for playlist analysis."""
        lines = response.split('\n')
        
        # Create mood analysis for playlist using neural network classification
        mood_analysis = MoodAnalysis(
            primary_mood=neural_mood,  # Use neural network classification
            secondary_mood="Multi-faceted",
            energy_level="Variable",
            emotional_tone="Neural Network Classified",
            mood_confidence=0.85,
            detailed_analysis=response
        )
        
        # Create playlist personality
        playlist_personality = PlaylistPersonality(
            personality_type="Curated Mix",
            listening_context=["General Listening", "Background Music"],
            mood_distribution={"happy": 0.3, "calm": 0.25, "energetic": 0.25, "melancholic": 0.2},
            genre_diversity="Moderate",
            overall_vibe="Balanced and engaging"
        )
        
        # Extract recommendations
        recommendations = []
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if "recommendation" in line.lower():
                in_recommendations = True
                continue
            
            if in_recommendations and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or 
                                      line.startswith(('-', '*'))):
                recommendation = {"title": line, "reason": "Complements playlist style"}
                recommendations.append(recommendation)
                if len(recommendations) >= 5:
                    break
        
        # Extract genre insights
        genre_insights = "Comprehensive analysis of genre diversity and musical cohesion."
        
        return AnalysisResult(
            mood_analysis=mood_analysis,
            genre_insights=genre_insights,
            recommendations=recommendations,
            playlist_personality=playlist_personality
        )
    
    def get_mood_recommendations(self, mood: str, energy_level: str) -> List[str]:
        """
        Get song recommendations based on mood and energy level.
        
        Args:
            mood: Target mood (happy, sad, calm, energetic, etc.)
            energy_level: Energy level (low, moderate, high)
            
        Returns:
            List of recommendation strings
        """
        prompt = f"""
Suggest 5 specific songs that match this mood and energy level:

- Mood: {mood}
- Energy Level: {energy_level}

For each song, provide:
- Artist
- Song Title (Year)
- Brief reason why it fits the mood

Focus on well-known tracks across different eras and genres.
"""
        
        messages = [
            {"role": "system", "content": "You are a music recommendation expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = self._call_groq_api(messages, temperature=0.8)
        return [line.strip() for line in response.split('\n') if line.strip()]
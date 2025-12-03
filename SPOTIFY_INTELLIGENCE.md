# Spotify Music Intelligence - Feature Documentation

## 🎧 Overview

The Spotify Music Intelligence feature is a powerful addition to the Music Mood Classification system that provides AI-powered analysis of Spotify tracks and playlists. It combines Spotify's Web API with advanced AI language models to generate comprehensive insights about music mood, genre characteristics, and personalized recommendations.

## ✨ Features

### 🎵 Track Analysis
- **Automatic URL Detection**: Supports both Spotify track and playlist URLs
- **Comprehensive Metadata**: Artist, album, genres, release date, popularity
- **Audio Feature Analysis**: Energy, valence, danceability, tempo, and more
- **AI-Powered Mood Analysis**: Generated using Groq LLM (Llama 3 or Mixtral)
- **Personalized Recommendations**: 5 similar tracks with detailed reasoning
- **Listening Context Suggestions**: Perfect scenarios for the track

### 🎶 Playlist Analysis  
- **Multi-Track Processing**: Analyze up to 100 tracks per playlist
- **Playlist Personality**: AI-generated personality type and characteristics
- **Mood Distribution**: Visual breakdown of emotional content
- **Genre Diversity Analysis**: Hierarchical genre categorization
- **Cohesion Analysis**: How well tracks flow together
- **Interactive Visualizations**: Mood landscape, genre distribution charts
- **Track-by-Track Insights**: Detailed analysis of individual songs

### 🔍 Advanced Visualizations
- **Radar Charts**: Audio feature profiles
- **Scatter Plots**: Energy vs. Valence mood landscapes
- **Pie Charts**: Genre and mood distributions
- **Bar Charts**: Category breakdowns
- **Interactive Tables**: Sortable track details

## 🛠️ Setup Instructions

### 1. API Credentials Setup

#### Spotify Web API
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app or use an existing one
3. Note your **Client ID** and **Client Secret**
4. Add `http://localhost:8501` to redirect URIs (if needed)

#### Groq AI API
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up or log in to your account
3. Generate an API key from the dashboard
4. Copy the API key for environment setup

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy from .env.example
cp .env.example .env
```

Edit `.env` with your credentials:
```env
SPOTIFY_CLIENT_ID=your_actual_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_actual_spotify_client_secret
GROQ_API_KEY=your_actual_groq_api_key
```

### 3. Install Dependencies

```bash
pip install requests python-dotenv
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 4. Load Environment Variables

The system automatically loads environment variables when you run the Streamlit app:

```bash
streamlit run app.py
```

## 📱 Usage Guide

### Analyzing a Single Track

1. **Navigate to Spotify Intelligence page**
2. **Paste a Spotify track URL**:
   ```
   https://open.spotify.com/track/4iV5W9uYEdYUVa79Axb7Rh
   ```
3. **Click "Analyze"**
4. **Review Results**:
   - Track metadata and audio features
   - AI-generated mood analysis
   - Genre insights and characteristics
   - Personalized recommendations
   - Listening context suggestions

### Analyzing a Playlist

1. **Paste a Spotify playlist URL**:
   ```
   https://open.spotify.com/playlist/37i9dQZF1DX0XUsuxWHRQd
   ```
2. **Set track limit** (10-100 tracks)
3. **Click "Analyze"**
4. **Explore Results**:
   - Playlist overview and metadata
   - AI-generated personality analysis
   - Mood and genre distribution charts
   - Track-by-track analysis
   - Cohesion and flow insights
   - Recommended additions

### Supported URL Formats

The system supports multiple Spotify URL formats:

- **Web URLs**: `https://open.spotify.com/track/[ID]`
- **Web URLs**: `https://open.spotify.com/playlist/[ID]`
- **Spotify URIs**: `spotify:track:[ID]`
- **Spotify URIs**: `spotify:playlist:[ID]`

## 🔧 Technical Architecture

### Module Structure
```
spotify_agent/
├── __init__.py           # Package initialization
├── spotify_api.py        # Spotify Web API client
├── analysis_agent.py     # Groq LLM analysis engine
└── utils.py              # Utility functions
```

### Key Components

#### `SpotifyClient`
- **Authentication**: Client Credentials OAuth flow
- **Data Fetching**: Tracks, playlists, audio features
- **Caching**: Token management and refresh
- **Error Handling**: Robust API error management

#### `MusicAnalysisAgent`
- **AI Analysis**: Groq LLM integration (Llama 3/Mixtral)
- **Prompt Engineering**: Structured prompts for music analysis
- **Response Parsing**: Extract insights from AI responses
- **Mood Classification**: Advanced emotional analysis

#### `Utils Module`
- **URL Validation**: Spotify URL format checking
- **Feature Formatting**: Human-readable audio features
- **Genre Hierarchy**: Intelligent genre categorization
- **Context Suggestions**: Listening scenario recommendations

### Data Flow

1. **URL Input** → Validation → ID Extraction
2. **Spotify API** → Fetch Metadata + Audio Features
3. **Data Processing** → Feature Calculation + Formatting
4. **AI Analysis** → Groq LLM → Mood/Genre Insights
5. **Visualization** → Charts + Interactive Elements
6. **Results Display** → Structured Presentation

### Caching Strategy

- **Streamlit Cache**: 1-hour TTL for API responses
- **Token Management**: Automatic refresh with 1-minute buffer
- **Rate Limiting**: Respectful API usage patterns

## 🎨 UI Components

### Interactive Elements
- **URL Input Field**: Real-time validation feedback
- **Analysis Button**: Progress indicators and status
- **Expandable Sections**: Organized information hierarchy
- **Tab Navigation**: Multiple analysis views

### Visualization Types
- **Radar Charts**: Multi-dimensional audio features
- **Scatter Plots**: 2D mood landscapes
- **Distribution Charts**: Genre and mood breakdowns
- **Data Tables**: Sortable track information
- **Progress Bars**: Feature intensity indicators

### Styling Features
- **Spotify Brand Colors**: Green gradient themes
- **Responsive Design**: Mobile-friendly layouts
- **Custom CSS**: Enhanced visual appearance
- **Interactive Tooltips**: Contextual information

## 🔒 Security & Privacy

### API Security
- **Environment Variables**: Credentials stored securely
- **No User Data Storage**: Analysis results not persisted
- **Rate Limiting Compliance**: Respects Spotify API limits
- **HTTPS Communication**: Secure API connections

### Privacy Considerations
- **No User Authentication**: Uses public data only
- **Anonymous Analysis**: No personal data required
- **Local Processing**: AI analysis via Groq API
- **No Data Retention**: Temporary analysis only

## 🚀 Performance Optimization

### Caching Mechanisms
- **API Response Caching**: Reduces redundant requests
- **Token Persistence**: Minimizes authentication overhead
- **Feature Computation**: Cached mathematical calculations

### Efficiency Features
- **Batch Processing**: Multiple tracks analyzed together
- **Lazy Loading**: Progressive data loading
- **Error Recovery**: Graceful degradation on failures
- **Timeout Handling**: Prevents hanging requests

## 🐛 Troubleshooting

### Common Issues

#### "Missing API Credentials"
- **Solution**: Ensure `.env` file exists with valid credentials
- **Check**: Environment variable names match exactly
- **Verify**: API keys are active and not expired

#### "Failed to Fetch Track/Playlist"
- **Solution**: Verify Spotify URL format is correct
- **Check**: Track/playlist is public (not private)
- **Retry**: Network connectivity and API status

#### "AI Analysis Failed"
- **Solution**: Check Groq API key validity
- **Fallback**: Basic analysis using Spotify features only
- **Alternative**: Try different Groq model

#### "Import Error"
- **Solution**: Install missing dependencies
- **Check**: Python path includes spotify_agent module
- **Reinstall**: `pip install -r requirements.txt`

### Performance Issues

#### Slow Analysis
- **Reduce**: Number of tracks analyzed (playlists)
- **Check**: Network connectivity speed
- **Monitor**: API rate limits and quotas

#### Memory Usage
- **Limit**: Playlist size to reasonable number
- **Clear**: Browser cache if using frequently
- **Restart**: Streamlit app if memory builds up

## 🔮 Future Enhancements

### Planned Features
- **Cross-Platform Support**: Apple Music, YouTube Music
- **Advanced Analytics**: Temporal mood analysis
- **User Preferences**: Personalized recommendation tuning
- **Export Options**: PDF reports, CSV data
- **Collaborative Features**: Shared analysis sessions

### Technical Improvements
- **Async Processing**: Faster playlist analysis
- **Database Integration**: Persistent analysis history
- **Advanced Caching**: Redis or similar backend
- **API Rate Optimization**: Smarter request batching
- **Mobile App**: Native mobile experience

## 📄 API Reference

### SpotifyClient Methods

```python
# Initialize client
client = SpotifyClient()

# Get track information
track = client.get_track(track_id)

# Get playlist with tracks
playlist = client.get_playlist(playlist_id, limit=50)

# Get audio features
features = client.get_audio_features([track_id])

# Search for tracks
results = client.search_track("query", limit=10)
```

### MusicAnalysisAgent Methods

```python
# Initialize agent
agent = MusicAnalysisAgent()

# Analyze single track
result = agent.analyze_track(track)

# Analyze playlist
result = agent.analyze_playlist(playlist)

# Compare track to playlist style
comparison = agent.compare_track_to_playlist_style(track, avg_features)

# Get mood-based recommendations
recs = agent.get_mood_recommendations("happy", "high")
```

### Utility Functions

```python
from spotify_agent.utils import *

# Detect URL type
url_type = detect_url_type(spotify_url)  # 'track' or 'playlist'

# Extract Spotify ID
spotify_id = extract_spotify_id(spotify_url)

# Format duration
duration_str = format_duration(225000)  # "3:45"

# Validate URL
is_valid, error = validate_spotify_url(url)

# Calculate mood from features
mood_info = calculate_mood_from_features(audio_features)
```

## 📞 Support

For issues, feature requests, or questions:

1. **GitHub Issues**: [Project Repository](https://github.com/viraj-gavade/music-mood-classification-ml/issues)
2. **Documentation**: This comprehensive guide
3. **Code Examples**: See `spotify_agent/` module for implementation details

## 📝 License

This feature is part of the Music Mood Classification ML project and follows the same licensing terms as the main project.

---

*The Spotify Music Intelligence feature enhances the core mood classification system with modern API integrations and AI-powered insights, providing a complete music analysis experience.*
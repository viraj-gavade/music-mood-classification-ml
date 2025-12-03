"""
Environment configuration loader for Spotify Intelligence
Ensures environment variables are loaded properly for the Streamlit app.
"""

import os
from pathlib import Path
import logging

def load_environment():
    """
    Load environment variables from .env file if available.
    This function is called automatically when the spotify_agent module is imported.
    """
    try:
        from dotenv import load_dotenv
        
        # Look for .env file in current directory and parent directories
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
            logging.info("Loaded environment variables from .env file")
        else:
            # Try parent directory
            parent_env = Path('../.env')
            if parent_env.exists():
                load_dotenv(parent_env)
                logging.info("Loaded environment variables from parent directory .env file")
            else:
                logging.warning("No .env file found. Make sure environment variables are set.")
        
    except ImportError:
        logging.warning("python-dotenv not installed. Environment variables must be set manually.")
    except Exception as e:
        logging.error(f"Failed to load environment variables: {e}")

def check_required_env_vars():
    """
    Check if all required environment variables are available.
    
    Returns:
        tuple: (success: bool, missing_vars: list)
    """
    required_vars = ['SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET', 'GROQ_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return len(missing_vars) == 0, missing_vars

def get_env_status():
    """
    Get comprehensive environment status for debugging.
    
    Returns:
        dict: Environment status information
    """
    success, missing = check_required_env_vars()
    
    return {
        'success': success,
        'missing_variables': missing,
        'spotify_configured': bool(os.getenv('SPOTIFY_CLIENT_ID') and os.getenv('SPOTIFY_CLIENT_SECRET')),
        'groq_configured': bool(os.getenv('GROQ_API_KEY')),
        'env_file_exists': Path('.env').exists(),
        'dotenv_available': True
    }

# Auto-load environment on import
try:
    load_environment()
except Exception as e:
    logging.error(f"Failed to auto-load environment: {e}")
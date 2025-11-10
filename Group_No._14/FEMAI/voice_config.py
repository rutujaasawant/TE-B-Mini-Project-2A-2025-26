"""
Voice Recognition Configuration
Supports Hindi, English, and Hinglish
"""
import os
from pathlib import Path

# Google Cloud credentials
GOOGLE_CREDENTIALS_PATH = os.environ.get(
    'GOOGLE_APPLICATION_CREDENTIALS',
    'google_Credential.json'  # UPDATE THIS
)

if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
    print(f"WARNING: Google credentials not found at {GOOGLE_CREDENTIALS_PATH}")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDENTIALS_PATH

# Language configurations
LANGUAGE_CONFIGS = {
    'hindi': {
        'code': 'hi-IN',
        'name': 'हिंदी',
        'enable_automatic_punctuation': True,
        'model': 'default',
        'use_enhanced': True
    },
    'english': {
        'code': 'en-US',
        'name': 'English',
        'enable_automatic_punctuation': True,
        'model': 'default',
        'use_enhanced': True
    },
    'hinglish': {
        'code': 'hi-IN',  # Use Hindi model for Hinglish
        'name': 'Hinglish',
        'enable_automatic_punctuation': True,
        'model': 'default',
        'use_enhanced': True,
        'alternative_language_codes': ['en-IN']  # Fallback to English-India
    }
}

# Audio recording settings
AUDIO_CONFIG = {
    'sample_rate': 16000,
    'chunk_size': 1024,
    'format': 'int16',
    'channels': 1,
    'max_duration': 10  # seconds
}

# Upload folder
VOICE_UPLOAD_FOLDER = 'voice_recordings'
os.makedirs(VOICE_UPLOAD_FOLDER, exist_ok=True)

# Allowed audio formats
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'}
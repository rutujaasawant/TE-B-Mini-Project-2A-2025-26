# FEMAI Health Application Configuration
# This file shows how to configure your application

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'femai-health-secret-key-2024')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # Chatbot Configuration
    CHATBOT_MODEL = os.getenv('CHATBOT_MODEL', 'gpt-3.5-turbo')
    CHATBOT_MAX_TOKENS = int(os.getenv('CHATBOT_MAX_TOKENS', 150))
    CHATBOT_TEMPERATURE = float(os.getenv('CHATBOT_TEMPERATURE', 0.7))
    
    # Hybrid Chatbot Settings
    USE_HYBRID_MODE = os.getenv('USE_HYBRID_MODE', 'True').lower() == 'true'
    API_COMPLEXITY_THRESHOLD = int(os.getenv('API_COMPLEXITY_THRESHOLD', 5))  # Words
    FORCE_API_FOR_CRUCIAL = os.getenv('FORCE_API_FOR_CRUCIAL', 'True').lower() == 'true'
    
    # Cost Optimization
    MAX_API_CALLS_PER_SESSION = int(os.getenv('MAX_API_CALLS_PER_SESSION', 50))
    API_COOLDOWN_SECONDS = int(os.getenv('API_COOLDOWN_SECONDS', 1))  # Rate limiting
    
    # Health API Configuration (for future integrations)
    HEALTH_API_ENABLED = os.getenv('HEALTH_API_ENABLED', 'True').lower() == 'true'
    HEALTH_API_KEY = os.getenv('HEALTH_API_KEY')
    
    # Voice and Speech API Configuration
    VOICE_API_ENABLED = os.getenv('VOICE_API_ENABLED', 'True').lower() == 'true'
    VOICE_API_PROVIDER = os.getenv('VOICE_API_PROVIDER', 'google')  # google, azure, amazon
    VOICE_API_KEY = os.getenv('VOICE_API_KEY')
    
    # Text-to-Speech API Configuration
    TTS_API_ENABLED = os.getenv('TTS_API_ENABLED', 'True').lower() == 'true'
    TTS_API_PROVIDER = os.getenv('TTS_API_PROVIDER', 'google')  # google, azure, amazon
    TTS_API_KEY = os.getenv('TTS_API_KEY')
    
    # Emergency Services API Configuration
    EMERGENCY_API_ENABLED = os.getenv('EMERGENCY_API_ENABLED', 'True').lower() == 'true'
    EMERGENCY_API_KEY = os.getenv('EMERGENCY_API_KEY')
    
    # Doctor Database Configuration
    DOCTORS_DATABASE = {
        "dr_sarah_johnson": {
            "id": "dr_sarah_johnson",
            "name": "Dr. Sarah Johnson",
            "specialty": "Gynecologist - PCOS Specialist",
            "phone": "+1-555-0123",
            "email": "dr.johnson@femaihealth.com",
            "available_hours": "Mon-Fri: 9:00 AM - 6:00 PM",
            "sat_hours": "Sat: 9:00 AM - 2:00 PM",
            "consultation_fee": "$150",
            "rating": 4.8,
            "experience": "15+ years",
            "languages": ["English", "Spanish"],
            "video_call_available": True,
            "voice_call_available": True,
            "status": "available"
        },
        "dr_emily_chen": {
            "id": "dr_emily_chen",
            "name": "Dr. Emily Chen",
            "specialty": "Endocrinologist - Hormone Specialist",
            "phone": "+1-555-0124",
            "email": "dr.chen@femaihealth.com",
            "available_hours": "Mon-Fri: 8:00 AM - 5:00 PM",
            "sat_hours": "Sat: 9:00 AM - 1:00 PM",
            "consultation_fee": "$180",
            "rating": 4.9,
            "experience": "12+ years",
            "languages": ["English", "Mandarin"],
            "video_call_available": True,
            "voice_call_available": True,
            "status": "available"
        },
        "dr_priya_patel": {
            "id": "dr_priya_patel",
            "name": "Dr. Priya Patel",
            "specialty": "Nutritionist - PCOS Diet Expert",
            "phone": "+1-555-0125",
            "email": "dr.patel@femaihealth.com",
            "available_hours": "Mon-Fri: 10:00 AM - 7:00 PM",
            "sat_hours": "Sat: 10:00 AM - 3:00 PM",
            "consultation_fee": "$120",
            "rating": 4.7,
            "experience": "8+ years",
            "languages": ["English", "Hindi", "Gujarati"],
            "video_call_available": True,
            "voice_call_available": True,
            "status": "available"
        },
        "dr_maria_rodriguez": {
            "id": "dr_maria_rodriguez",
            "name": "Dr. Maria Rodriguez",
            "specialty": "Fertility Specialist - PCOS Expert",
            "phone": "+1-555-0126",
            "email": "dr.rodriguez@femaihealth.com",
            "available_hours": "Mon-Fri: 9:00 AM - 6:00 PM",
            "sat_hours": "Sat: 9:00 AM - 2:00 PM",
            "consultation_fee": "$200",
            "rating": 4.9,
            "experience": "18+ years",
            "languages": ["English", "Spanish", "Portuguese"],
            "video_call_available": True,
            "voice_call_available": True,
            "status": "available"
        }
    }
    
    # Call Configuration
    CALL_TIMEOUT_SECONDS = int(os.getenv('CALL_TIMEOUT_SECONDS', 300))  # 5 minutes
    MAX_CALL_DURATION = int(os.getenv('MAX_CALL_DURATION', 3600))  # 1 hour
    CALL_RECORDING_ENABLED = os.getenv('CALL_RECORDING_ENABLED', 'True').lower() == 'true'
    
    # API Rate Limiting
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', 100))  # requests per hour
    API_COOLDOWN_SECONDS = int(os.getenv('API_COOLDOWN_SECONDS', 1))
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @staticmethod
    def validate_config():
        """Validate that required configuration is present"""
        missing_configs = []
        
        if not Config.OPENAI_API_KEY:
            missing_configs.append("OPENAI_API_KEY")
        
        if missing_configs:
            print(f"Warning: Missing configuration for: {', '.join(missing_configs)}")
            print("The application will run in fallback mode.")
            print("\nTo enable AI-powered responses:")
            print("1. Get an OpenAI API key from: https://platform.openai.com/api-keys")
            print("2. Create a .env file in your project root")
            print("3. Add: OPENAI_API_KEY=your_actual_api_key_here")
            print("\nTo create a .env file:")
            print("1. Create a new file named '.env' (with the dot)")
            print("2. Add your configuration variables")
            print("3. Never commit this file to version control")
        
        return len(missing_configs) == 0

# Configuration validation on import
Config.validate_config()

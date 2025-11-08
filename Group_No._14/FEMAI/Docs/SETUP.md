# FEMAI Health Application Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API (Optional but Recommended)

#### Option A: Using Environment Variables
Create a `.env` file in your project root:
```bash
# Create .env file
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

#### Option B: Using config.py
Edit `config.py` and set your API key directly:
```python
OPENAI_API_KEY = "your_actual_api_key_here"
```

### 3. Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key and add it to your `.env` file

### 4. Run the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

## Configuration Options

### Environment Variables (.env file)
```bash
# Required for AI-powered responses
OPENAI_API_KEY=your_openai_api_key_here

# Optional Flask settings
SECRET_KEY=your_secret_key_here
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000

# Optional Chatbot settings
CHATBOT_MODEL=gpt-3.5-turbo
CHATBOT_MAX_TOKENS=150
CHATBOT_TEMPERATURE=0.7

# Hybrid Chatbot Settings
USE_HYBRID_MODE=True
API_COMPLEXITY_THRESHOLD=5
FORCE_API_FOR_CRUCIAL=True
MAX_API_CALLS_PER_SESSION=50
API_COOLDOWN_SECONDS=1
```

### Configuration File (config.py)
All settings can also be configured in `config.py`:
```python
class Config:
    OPENAI_API_KEY = "your_api_key_here"
    CHATBOT_MODEL = "gpt-4"  # Use GPT-4 for better responses
    CHATBOT_MAX_TOKENS = 200  # Allow longer responses
    CHATBOT_TEMPERATURE = 0.8  # More creative responses
    
    # Hybrid Chatbot Settings
    USE_HYBRID_MODE = True  # Enable smart API/local response selection
    API_COMPLEXITY_THRESHOLD = 5  # Use API for messages with >5 words
    FORCE_API_FOR_CRUCIAL = True  # Always use API for health topics
    MAX_API_CALLS_PER_SESSION = 50  # Limit API usage per session
    API_COOLDOWN_SECONDS = 1  # Rate limiting between API calls
```

### Hybrid Mode Benefits
- **Cost Control**: Only pay for complex queries that need AI
- **Speed**: Basic questions get instant responses
- **Intelligence**: Complex health questions get personalized AI answers
- **Reliability**: Works offline for simple queries
- **Scalability**: Easy to adjust API usage based on needs

## How the Chatbot Works

### Hybrid Approach (Recommended)
Your chatbot now uses a **smart hybrid system** that optimizes both cost and user experience:

#### Local Responses (Basic Queries)
- **Greetings**: "Hello", "Hi", "Good morning"
- **Simple acknowledgments**: "Yes", "No", "Thanks"
- **Emergency information**: Immediate responses for urgent situations
- **Response time**: Instant (0-50ms)

#### OpenAI API Responses (Crucial Queries)
- **Health symptoms**: Detailed symptom explanations
- **Diet recommendations**: Personalized nutrition advice
- **Exercise plans**: Customized fitness recommendations
- **Treatment options**: Up-to-date medical information
- **Lifestyle advice**: Personalized wellness tips
- **Complex questions**: "How to", "Why", "What if" scenarios
- **Response time**: 1-3 seconds

### Smart Decision Making
The chatbot automatically decides when to use API based on:
- **Message complexity**: Longer questions get API responses
- **Question type**: "How", "Why", "What if" trigger API
- **Health topics**: Symptoms, diet, exercise, medication use API
- **User intent**: Seeking advice or detailed information

### Cost Optimization
- **Basic queries**: 0 API calls (free)
- **Complex queries**: 1 API call per question
- **Rate limiting**: Built-in cooldown to prevent abuse
- **Session limits**: Configurable maximum API calls per session

### Fallback Mode (No API Key)
- **Always available**: Works without internet or API key
- **Reliable responses**: Pre-defined answers for common topics
- **Limited flexibility**: Static responses, no personalization

## Testing the Chatbot

### 1. Start the application
```bash
python app.py
```

### 2. Open your browser
Navigate to `http://localhost:5000/chatbot`

### 3. Test with sample questions
- "What are the symptoms of PCOS?"
- "What should I eat to manage PCOD?"
- "How much exercise should I do?"
- "What medications are available?"

### 4. Check the console
You'll see logs showing whether responses come from OpenAI API or fallback system.

## Troubleshooting

### "No OpenAI API key found"
- Create a `.env` file with your API key
- Or edit `config.py` directly
- The app will still work in fallback mode

### "OpenAI API failed"
- Check your internet connection
- Verify your API key is correct
- Check OpenAI service status
- The app will automatically use fallback responses

### "Module not found"
- Run `pip install -r requirements.txt`
- Make sure you're in the correct directory

## Security Notes

- **Never commit your `.env` file** to version control
- **Keep your API key private** and secure
- **Monitor API usage** to avoid unexpected charges
- **Use environment variables** in production deployments

## Next Steps

1. **Customize Responses**: Edit the fallback responses in `app.py`
2. **Add More Features**: Integrate with health databases, add user accounts
3. **Deploy**: Use services like Heroku, AWS, or DigitalOcean
4. **Scale**: Add database, user management, and analytics

## Support

If you encounter issues:
1. Check the console logs for error messages
2. Verify your configuration is correct
3. Test with a simple message first
4. Ensure all dependencies are installed

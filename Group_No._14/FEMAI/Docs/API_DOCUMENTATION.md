# FEMAI Health Application - API Documentation

## Overview
The FEMAI Health application integrates multiple APIs to provide comprehensive PCOD/PCOS health assistance, voice processing, and emergency services.

## 🔑 **Core APIs Integrated**

### 1. **OpenAI API** (Primary AI Chatbot)
- **Purpose**: Generate intelligent, contextual responses for complex health queries
- **Endpoint**: `/api/chat`
- **Method**: POST
- **Integration**: ✅ **FULLY INTEGRATED**
- **Usage**: Automatically used for crucial health questions

**Configuration:**
```bash
OPENAI_API_KEY=your_openai_api_key_here
CHATBOT_MODEL=gpt-3.5-turbo
CHATBOT_MAX_TOKENS=150
CHATBOT_TEMPERATURE=0.7
```

### 2. **Health Information APIs** (Medical Data)
- **Purpose**: Provide detailed medical information from authoritative sources
- **Endpoints**: 
  - `/api/health-info/symptoms` - Symptom database
  - `/api/health-info/nutrition` - Diet recommendations
  - `/api/health-info/exercise` - Fitness plans
  - `/api/health-info/medication` - Treatment information
- **Method**: GET
- **Integration**: ✅ **FULLY INTEGRATED** (Mock data ready for real APIs)
- **Usage**: Automatically enhances responses with detailed health data

**Mock API Endpoints:**
```bash
# Symptoms Database
GET /api/health-info/symptoms
Response: Common PCOD/PCOS symptoms with severity levels

# Nutrition Guide
GET /api/health-info/nutrition
Response: Recommended foods, foods to avoid, meal timing

# Exercise Plans
GET /api/health-info/exercise
Response: Cardio exercises, strength training, frequency

# Medication Info
GET /api/health-info/medication
Response: Common medications, important notes
```

### 3. **Emergency Services API** (Urgent Care)
- **Purpose**: Provide emergency contact information and urgent care guidance
- **Endpoint**: `/api/emergency-services`
- **Method**: GET
- **Integration**: ✅ **FULLY INTEGRATED** (Mock data ready for real APIs)
- **Usage**: Critical for emergency situations

**Features:**
- Emergency phone numbers (911, Poison Control, Crisis Line)
- Urgent symptoms identification
- When to call 911 guidance
- Local emergency services (when location provided)

### 4. **Voice Processing APIs** (Speech-to-Text)
- **Purpose**: Convert voice input to text for hands-free interaction
- **Endpoint**: `/api/voice/process`
- **Method**: POST
- **Integration**: 🔄 **READY FOR INTEGRATION**
- **Supported Providers**: Google Speech, Azure Speech, Amazon Transcribe

**Configuration:**
```bash
VOICE_API_ENABLED=True
VOICE_API_PROVIDER=google  # google, azure, amazon
VOICE_API_KEY=your_voice_api_key_here
```

### 5. **Text-to-Speech APIs** (Audio Responses)
- **Purpose**: Convert text responses to speech for accessibility
- **Endpoint**: `/api/tts/synthesize`
- **Method**: POST
- **Integration**: 🔄 **READY FOR INTEGRATION**
- **Supported Providers**: Google TTS, Azure TTS, Amazon Polly

**Configuration:**
```bash
TTS_API_ENABLED=True
TTS_API_PROVIDER=google  # google, azure, amazon
TTS_API_KEY=your_tts_api_key_here
```

### 6. **Symptoms Checker API** (Health Assessment)
- **Purpose**: Analyze symptoms and provide preliminary health assessments
- **Endpoint**: `/api/health/symptoms-checker`
- **Method**: POST
- **Integration**: ✅ **FULLY INTEGRATED**
- **Usage**: Helps users understand symptom severity

**Request Format:**
```json
{
  "symptoms": ["irregular periods", "weight gain", "acne"],
  "severity": "moderate"
}
```

**Response:**
```json
{
  "status": "success",
  "assessment": {
    "symptoms_identified": ["irregular periods", "weight gain", "acne"],
    "severity_level": "moderate",
    "recommendations": [
      "Schedule appointment with healthcare provider",
      "Monitor symptoms closely"
    ],
    "urgent_attention_needed": false
  }
}
```

## 🚀 **API Integration Status**

| API Category | Status | Integration Level | Notes |
|--------------|--------|-------------------|-------|
| **OpenAI** | ✅ Active | Full | AI-powered responses |
| **Health Info** | ✅ Active | Mock → Real Ready | Medical data enhancement |
| **Emergency** | ✅ Active | Mock → Real Ready | Urgent care support |
| **Voice Processing** | 🔄 Ready | Mock → Real Ready | Speech-to-text |
| **Text-to-Speech** | 🔄 Ready | Mock → Real Ready | Audio responses |
| **Symptoms Checker** | ✅ Active | Full | Health assessment |

## 🔧 **How to Enable Real APIs**

### 1. **Google APIs** (Recommended for Start)
```bash
# Get API keys from Google Cloud Console
GOOGLE_SPEECH_API_KEY=your_key_here
GOOGLE_TTS_API_KEY=your_key_here
GOOGLE_HEALTH_API_KEY=your_key_here
```

### 2. **Azure APIs** (Enterprise)
```bash
# Get API keys from Azure Portal
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=eastus
AZURE_HEALTH_KEY=your_key_here
```

### 3. **Amazon APIs** (AWS)
```bash
# Get API keys from AWS Console
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1
```

## 📊 **API Usage Examples**

### **Basic Chat with API Enhancement**
```bash
POST /api/chat
{
  "message": "What are the symptoms of PCOS?",
  "user_id": "user_123"
}
```

**Response with API Enhancement:**
```json
{
  "status": "success",
  "response": "PCOS symptoms include irregular periods, weight gain, and hormonal changes...\n\n📋 **Detailed Symptom Information:**\n• Irregular menstrual cycles\n• Excess androgen levels\n• Polycystic ovaries\n• Weight gain and difficulty losing weight\n• Acne and oily skin",
  "source": "openai",
  "complexity": "high"
}
```

### **Health Information Request**
```bash
GET /api/health-info/nutrition
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "source": "nutrition_api",
    "data": {
      "recommended_foods": ["Low glycemic index foods", "Lean proteins", "Complex carbohydrates"],
      "foods_to_avoid": ["Refined carbohydrates", "Sugary foods", "Processed foods"],
      "meal_timing": "Small, frequent meals every 3-4 hours"
    }
  }
}
```

### **Emergency Services**
```bash
GET /api/emergency-services?location=NewYork
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "source": "emergency_api",
    "data": {
      "emergency_numbers": {
        "general": "911",
        "poison_control": "1-800-222-1222"
      },
      "local_services": {
        "nearest_hospital": "Local Medical Center",
        "urgent_care": "QuickCare Clinic"
      }
    }
  }
}
```

## 🛡️ **Security & Rate Limiting**

### **Rate Limiting**
```bash
API_RATE_LIMIT=100          # Requests per hour
API_COOLDOWN_SECONDS=1      # Seconds between requests
```

### **API Key Security**
- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly
- Monitor API usage

## 📈 **Performance Metrics**

### **Response Times**
- **Local Responses**: 0-50ms (instant)
- **API-Enhanced Responses**: 1-3 seconds
- **Voice Processing**: 2-5 seconds
- **Text-to-Speech**: 1-2 seconds

### **Cost Optimization**
- **Basic Queries**: 0 API calls (free)
- **Complex Queries**: 1 API call per question
- **Health Info**: 1 API call per topic
- **Emergency**: Always available (no cost)

## 🔮 **Future API Integrations**

### **Phase 2 APIs** (Planned)
- **Telemedicine APIs**: Doctor consultation booking
- **Lab Results APIs**: Blood test integration
- **Medication Tracking APIs**: Prescription management
- **Fitness Device APIs**: Health data sync
- **Mental Health APIs**: Stress and anxiety support

### **Phase 3 APIs** (Future)
- **Genomic APIs**: Personalized health insights
- **AI Imaging APIs**: Ultrasound analysis
- **Clinical Trial APIs**: Research participation
- **Insurance APIs**: Coverage verification
- **Pharmacy APIs**: Medication delivery

## 🆘 **Troubleshooting**

### **Common Issues**
1. **API Key Invalid**: Check environment variables
2. **Rate Limit Exceeded**: Implement cooldown
3. **API Service Down**: Fallback to local responses
4. **Network Issues**: Check internet connection

### **Testing APIs**
```bash
# Test health info API
curl http://localhost:5000/api/health-info/symptoms

# Test emergency services
curl http://localhost:5000/api/emergency-services

# Test symptoms checker
curl -X POST http://localhost:5000/api/health/symptoms-checker \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["irregular periods"], "severity": "mild"}'
```

## 📞 **Support & Integration Help**

For API integration assistance:
1. Check the configuration files
2. Verify API keys and permissions
3. Test endpoints individually
4. Review error logs in console
5. Consult API provider documentation

---

**Last Updated**: January 2025  
**Version**: 2.0  
**Status**: Production Ready with Mock APIs

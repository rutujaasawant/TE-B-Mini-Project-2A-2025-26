# FEMAI-Health: PCOD/PCOS Health Assistant

A comprehensive web application designed to support women with PCOD/PCOS through an intelligent chatbot, voice/video consultations, and comprehensive health information. Built with Flask and modern web technologies, this application is designed to be accessible even for illiterate users.

## 🌟 Features

### 🤖 Intelligent Chatbot Assistant
- **24/7 Health Support**: Get instant answers to PCOD/PCOS questions
- **Voice Input**: Speak your questions instead of typing
- **Text-to-Speech**: Listen to responses for better understanding
- **Quick Topic Buttons**: Easy access to common health topics
- **Multilingual Support**: Designed for diverse user populations

### 📞 Voice Call Integration
- **Healthcare Professional Calls**: Connect with doctors, nurses, and specialists
- **Call Quality Monitoring**: Real-time audio quality indicators
- **Call Recording**: Save consultations for medical records
- **Emergency Support**: Quick access to emergency services
- **Call History**: Track all your consultations

### 📹 Video Consultation
- **Face-to-Face Consultations**: Visual communication with healthcare experts
- **Screen Sharing**: Share documents, test results, or symptoms
- **High-Quality Video**: Adjustable video quality settings
- **Device Management**: Camera, microphone, and speaker controls
- **Recording Capability**: Save video consultations

### 📚 Comprehensive Health Information
- **PCOD/PCOS Education**: Complete guide to understanding the condition
- **Symptom Tracker**: Monitor your health patterns over time
- **Treatment Options**: Information about medications and therapies
- **Lifestyle Management**: Diet, exercise, and stress management tips
- **Emergency Guidelines**: When to seek immediate medical help

### ♿ Accessibility Features
- **Large Buttons**: Easy-to-use interface for all users
- **Voice Commands**: Control the application with voice
- **High Contrast**: Support for users with visual impairments
- **Keyboard Navigation**: Full keyboard accessibility
- **Screen Reader Support**: Compatible with assistive technologies

## 🚀 Technology Stack

### Backend
- **Flask**: Python web framework
- **Flask-SocketIO**: Real-time communication
- **Python 3.8+**: Core programming language

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **JavaScript (ES6+)**: Interactive functionality
- **WebRTC**: Voice and video communication
- **Socket.IO**: Real-time features

### APIs & Integration
- **Health Data APIs**: Integration with medical databases
- **Speech Recognition**: Voice input processing
- **Text-to-Speech**: Audio response generation
- **Media Devices**: Camera and microphone access

## 📋 Prerequisites

Before running this application, ensure you have:

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **Microphone and camera** (for voice/video features)
- **Stable internet connection**

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/femai-health.git
cd femai-health
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a `.env` file in the root directory:
```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

### 5. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 🎯 Usage Guide

### Getting Started
1. **Open the Application**: Navigate to the homepage
2. **Choose Your Need**: Select from chatbot, voice call, video call, or health info
3. **Start Interacting**: Use the intuitive interface to get help

### Using the Chatbot
1. **Ask Questions**: Type or speak your health questions
2. **Quick Topics**: Use the topic buttons for common questions
3. **Voice Input**: Click the microphone button to speak
4. **Listen to Responses**: Enable text-to-speech for audio responses

### Making Voice Calls
1. **Select Contact**: Choose the type of healthcare professional
2. **Start Call**: Click the large call button
3. **Call Controls**: Use mute, hold, and other features during calls
4. **End Call**: Click the end call button when finished

### Video Consultations
1. **Enable Camera**: Allow camera and microphone access
2. **Start Video Call**: Begin your consultation
3. **Screen Share**: Share documents or test results
4. **Quality Settings**: Adjust video and audio quality as needed

### Health Information
1. **Browse Topics**: Explore symptoms, treatments, and lifestyle tips
2. **Symptom Tracker**: Log your daily health patterns
3. **Emergency Info**: Access emergency contact numbers
4. **Resources**: Find support groups and educational materials

## 🔧 Configuration

### Chatbot Settings
- **Response Language**: Configure language preferences
- **Voice Settings**: Adjust speech recognition sensitivity
- **Topic Customization**: Modify quick topic buttons

### Call Settings
- **Audio Quality**: Set preferred call quality
- **Recording**: Enable/disable call recording
- **Notifications**: Configure call alerts

### Video Settings
- **Resolution**: Choose video quality
- **Bandwidth**: Adjust for connection speed
- **Device Selection**: Choose camera and microphone

## 🌐 API Endpoints

### Chat API
- `POST /api/chat` - Send messages to chatbot
- `GET /api/health-data` - Retrieve health information

### Socket.IO Events
- `connect` - Client connection
- `voice_message` - Voice communication
- `video_frame` - Video streaming
- `join_room` - Join call rooms

## 🔒 Security Features

- **HTTPS Support**: Secure communication
- **User Authentication**: Secure access control
- **Data Encryption**: Protect sensitive health information
- **Privacy Controls**: User consent for data collection
- **Audit Logging**: Track system access and usage

## 📱 Mobile Responsiveness

The application is fully responsive and works on:
- **Desktop computers**
- **Tablets**
- **Mobile phones**
- **All modern browsers**

## 🧪 Testing

### Run Tests
```bash
python -m pytest tests/
```

### Test Coverage
```bash
python -m pytest --cov=app tests/
```

## 🚀 Deployment

### Production Deployment
1. **Set Environment Variables**
2. **Configure Web Server** (Nginx/Apache)
3. **Set up SSL Certificate**
4. **Configure Database** (if using external storage)
5. **Set up Monitoring**

### Docker Deployment
```bash
docker build -t femai-health .
docker run -p 5000:5000 femai-health
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Documentation**: Check this README and code comments
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact the development team

### Emergency Support
For medical emergencies, please contact your local emergency services immediately. This application is for informational purposes only and should not replace professional medical advice.

## 🙏 Acknowledgments

- **Healthcare Professionals**: For medical guidance and validation
- **Open Source Community**: For the amazing tools and libraries
- **User Community**: For feedback and suggestions
- **Research Institutions**: For medical research and data

## 📊 Project Status

- **Current Version**: 1.0.0
- **Development Status**: Active Development
- **Last Updated**: December 2024
- **Next Release**: Q1 2025

## 🔮 Future Roadmap

### Phase 2 (Q1 2025)
- [ ] AI-powered symptom analysis
- [ ] Integration with wearable devices
- [ ] Telemedicine platform integration
- [ ] Multi-language support

### Phase 3 (Q2 2025)
- [ ] Mobile app development
- [ ] Advanced analytics dashboard
- [ ] Machine learning improvements
- [ ] Community features

### Phase 4 (Q3 2025)
- [ ] Virtual reality consultations
- [ ] Blockchain health records
- [ ] Advanced AI diagnostics
- [ ] Global expansion

---

**Made with ❤️ for women's health and empowerment**

For more information, visit our website or contact the development team.

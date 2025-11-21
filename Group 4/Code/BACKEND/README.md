# Smart Home Energy Management System

A comprehensive full-stack application for monitoring, predicting, and optimizing home energy consumption using ARIMA forecasting models.

## Features

### Frontend
- **Modern Web Interface**: Responsive HTML/CSS/JavaScript frontend with dark/light mode
- **Real-time Dashboard**: Live energy consumption monitoring and device management
- **Interactive Charts**: Energy consumption trends visualization using Chart.js
- **Device Management**: Add, remove, and control smart home devices
- **User Authentication**: Secure login/signup system
- **Energy Predictions**: ARIMA-based consumption forecasting
- **Energy Saving Tips**: Personalized recommendations for efficiency

### Backend
- **FastAPI Framework**: High-performance Python REST API
- **JWT Authentication**: Secure token-based user authentication
- **CSV Data Storage**: Lightweight data persistence with CSV files
- **ARIMA Modeling**: Statistical forecasting using statsmodels
- **Real-time Analytics**: Live energy consumption statistics
- **Device Control**: RESTful APIs for device management
- **Cross-platform**: Docker support for easy deployment

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js (for frontend development)
- Git

### Installation

1. **Clone the repository**
\`\`\`bash
git clone <repository-url>
cd smart-energy-backend
\`\`\`

2. **Backend Setup**
\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python scripts/generate_sample_data.py

# Start the backend server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
\`\`\`

3. **Frontend Setup**
\`\`\`bash
# Open frontend/index.html in your browser
# Or serve it using a simple HTTP server
python -m http.server 3000 --directory frontend
\`\`\`

### Using Docker

\`\`\`bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application
# Backend API: http://localhost:8000
# Frontend: http://localhost:3000
\`\`\`

## API Documentation

### Authentication Endpoints

#### POST /auth/signup
Register a new user
\`\`\`json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "password123"
}
\`\`\`

#### POST /auth/login
Login existing user
\`\`\`json
{
  "email": "john@example.com",
  "password": "password123"
}
\`\`\`

### Device Management

#### GET /devices
Get all user devices (requires authentication)

#### POST /devices
Create a new device
\`\`\`json
{
  "name": "Living Room AC",
  "type": "ac",
  "power": 1500,
  "status": "off"
}
\`\`\`

#### PUT /devices/{device_id}
Update device status or consumption
\`\`\`json
{
  "status": "on",
  "consumption": 12.5
}
\`\`\`

#### DELETE /devices/{device_id}
Delete a device

### Analytics Endpoints

#### GET /analytics/statistics
Get energy consumption statistics

#### GET /analytics/consumption
Get consumption data and daily trends

#### POST /analytics/predict
Generate ARIMA energy consumption prediction
\`\`\`json
{
  "days": 30
}
\`\`\`

## Sample Data

The system comes with pre-generated sample data:

### Test Users
- **Email**: john@example.com, **Password**: password123
- **Email**: jane@example.com, **Password**: password123
- **Email**: mike@example.com, **Password**: password123

### Device Types
- Air Conditioners (1200-2500W)
- Heaters (1000-3500W)
- Lights (20-150W)
- Kitchen Appliances (100-2000W)
- Entertainment Systems (50-500W)

## Data Structure

### CSV Files
- `users.csv`: User accounts and authentication
- `devices.csv`: Smart home devices and their properties
- `energy_data.csv`: Historical energy consumption records
- `device_types.csv`: Device categories and specifications
- `energy_rates.csv`: Regional electricity pricing
- `weather_data.csv`: Weather data for consumption correlation

## ARIMA Prediction Model

The system uses ARIMA (AutoRegressive Integrated Moving Average) models for energy consumption forecasting:

- **Model Order**: ARIMA(2,1,2) for optimal balance of accuracy and performance
- **Seasonal Factors**: Weekly and daily consumption patterns
- **Weather Integration**: Temperature-based consumption adjustments
- **Trend Analysis**: Long-term consumption pattern recognition

## Architecture

\`\`\`
Frontend (HTML/CSS/JS)
    ↓ HTTP/REST API
Backend (FastAPI/Python)
    ↓ File I/O
Data Storage (CSV Files)
    ↓ Statistical Analysis
ARIMA Model (statsmodels)
\`\`\`

## Development

### Adding New Features

1. **Backend**: Add new endpoints in `main.py`
2. **Frontend**: Update `frontend/index.html` with new UI components
3. **Data**: Modify CSV structure and update data models
4. **Prediction**: Enhance ARIMA model in prediction endpoint

### Testing

\`\`\`bash
# Validate data integrity
python scripts/data_validator.py

# Test API endpoints
curl -X GET http://localhost:8000/health

# Generate statistics
python scripts/data_validator.py
\`\`\`

## Deployment

### Production Setup

1. **Environment Variables**
\`\`\`bash
export SECRET_KEY="your-production-secret-key"
export DATABASE_URL="your-database-connection-string"
\`\`\`

2. **Docker Production**
\`\`\`bash
docker build -t smart-energy-app .
docker run -p 8000:8000 -v ./data:/app/data smart-energy-app
\`\`\`

3. **Reverse Proxy** (Nginx)
\`\`\`nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location /api/ {
        proxy_pass http://localhost:8000/;
    }
    
    location / {
        root /path/to/frontend;
        try_files $uri $uri/ /index.html;
    }
}
\`\`\`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check the API documentation at `/docs` endpoint
- Review the data validation scripts for troubleshooting

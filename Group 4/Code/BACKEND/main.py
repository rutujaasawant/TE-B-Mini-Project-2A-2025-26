from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import jwt
import os
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Smart Energy Management API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"

# Data models
class User(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Device(BaseModel):
    name: str
    type: str
    power: int
    status: str = "off"

class DeviceUpdate(BaseModel):
    status: Optional[str] = None
    consumption: Optional[float] = None

class PredictionRequest(BaseModel):
    days: int = 30

# Global variables for data storage
users_df = pd.DataFrame()
devices_df = pd.DataFrame()
energy_data_df = pd.DataFrame()

def load_data():
    """Load data from CSV files"""
    global users_df, devices_df, energy_data_df
    try:
        users_df = pd.read_csv('data/users.csv')
    except FileNotFoundError:
        users_df = pd.DataFrame(columns=['id', 'name', 'email', 'password_hash', 'avatar'])
    
    try:
        devices_df = pd.read_csv('data/devices.csv')
    except FileNotFoundError:
        devices_df = pd.DataFrame(columns=['id', 'user_id', 'name', 'type', 'power', 'status', 'consumption', 'created_at'])
    
    try:
        energy_data_df = pd.read_csv('data/energy_data.csv')
        energy_data_df['timestamp'] = pd.to_datetime(energy_data_df['timestamp'])
    except FileNotFoundError:
        energy_data_df = pd.DataFrame(columns=['id', 'user_id', 'device_id', 'consumption', 'timestamp'])

def save_data():
    """Save data to CSV files"""
    os.makedirs('data', exist_ok=True)
    users_df.to_csv('data/users.csv', index=False)
    devices_df.to_csv('data/devices.csv', index=False)
    energy_data_df.to_csv('data/energy_data.csv', index=False)

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Initialize data on startup
@app.on_event("startup")
async def startup_event():
    load_data()

# Authentication endpoints
@app.post("/auth/signup")
async def signup(user: User):
    global users_df
    
    # Check if user already exists
    if not users_df.empty and user.email in users_df['email'].values:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create new user
    user_id = len(users_df) + 1
    new_user = {
        'id': user_id,
        'name': user.name,
        'email': user.email,
        'password_hash': hash_password(user.password),
        'avatar': user.name[0].upper()
    }
    
    users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
    save_data()
    
    # Create access token
    access_token = create_access_token({"user_id": user_id})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": user.name,
            "email": user.email,
            "avatar": user.name[0].upper()
        }
    }

@app.post("/auth/login")
async def login(user_login: UserLogin):
    # Find user
    if users_df.empty:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_row = users_df[users_df['email'] == user_login.email]
    if user_row.empty:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_data = user_row.iloc[0]
    
    # Verify password
    if not verify_password(user_login.password, user_data['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create access token
    access_token = create_access_token({"user_id": int(user_data['id'])})
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": int(user_data['id']),
            "name": user_data['name'],
            "email": user_data['email'],
            "avatar": user_data['avatar']
        }
    }

# Device management endpoints
@app.get("/devices")
async def get_devices(current_user: int = Depends(get_current_user)):
    if devices_df.empty:
        return []
    
    user_devices = devices_df[devices_df['user_id'] == current_user]
    return user_devices.to_dict('records')

@app.post("/devices")
async def create_device(device: Device, current_user: int = Depends(get_current_user)):
    global devices_df
    
    device_id = len(devices_df) + 1
    new_device = {
        'id': device_id,
        'user_id': current_user,
        'name': device.name,
        'type': device.type,
        'power': device.power,
        'status': device.status,
        'consumption': 0.0,
        'created_at': datetime.now().isoformat()
    }
    
    devices_df = pd.concat([devices_df, pd.DataFrame([new_device])], ignore_index=True)
    save_data()
    
    return new_device

@app.put("/devices/{device_id}")
async def update_device(device_id: int, device_update: DeviceUpdate, current_user: int = Depends(get_current_user)):
    global devices_df
    
    # Find device
    device_mask = (devices_df['id'] == device_id) & (devices_df['user_id'] == current_user)
    if not device_mask.any():
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Update device
    if device_update.status is not None:
        devices_df.loc[device_mask, 'status'] = device_update.status
        # Calculate consumption based on status
        if device_update.status == 'on':
            power = devices_df.loc[device_mask, 'power'].iloc[0]
            consumption = (power / 1000) * (8 + np.random.random() * 8)
            devices_df.loc[device_mask, 'consumption'] = consumption
        else:
            devices_df.loc[device_mask, 'consumption'] = 0.0
    
    if device_update.consumption is not None:
        devices_df.loc[device_mask, 'consumption'] = device_update.consumption
    
    save_data()
    
    updated_device = devices_df[device_mask].iloc[0].to_dict()
    return updated_device

@app.delete("/devices/{device_id}")
async def delete_device(device_id: int, current_user: int = Depends(get_current_user)):
    global devices_df
    
    # Find device
    device_mask = (devices_df['id'] == device_id) & (devices_df['user_id'] == current_user)
    if not device_mask.any():
        raise HTTPException(status_code=404, detail="Device not found")
    
    # Delete device
    devices_df = devices_df[~device_mask]
    save_data()
    
    return {"message": "Device deleted successfully"}

# Energy analytics endpoints
@app.get("/analytics/consumption")
async def get_consumption_data(current_user: int = Depends(get_current_user)):
    if devices_df.empty:
        return {"total_consumption": 0, "active_devices": 0, "daily_data": []}
    
    user_devices = devices_df[devices_df['user_id'] == current_user]
    total_consumption = user_devices['consumption'].sum()
    active_devices = len(user_devices[user_devices['status'] == 'on'])
    
    # Generate daily data for the last 30 days
    daily_data = []
    for i in range(30):
        date = datetime.now() - timedelta(days=29-i)
        consumption = total_consumption * (0.8 + np.random.random() * 0.4)
        daily_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "consumption": round(consumption, 2)
        })
    
    return {
        "total_consumption": round(total_consumption, 2),
        "active_devices": active_devices,
        "daily_data": daily_data
    }

@app.get("/analytics/statistics")
async def get_statistics(current_user: int = Depends(get_current_user)):
    if devices_df.empty:
        return {
            "total_consumption": 0,
            "monthly_cost": 0,
            "active_devices": 0,
            "efficiency": 0
        }
    
    user_devices = devices_df[devices_df['user_id'] == current_user]
    total_consumption = user_devices['consumption'].sum()
    active_devices = len(user_devices[user_devices['status'] == 'on'])
    
    # Calculate costs (assuming ₹8 per kWh)
    electricity_rate = 8.0
    daily_cost = total_consumption * electricity_rate
    monthly_cost = daily_cost * 30
    
    # Calculate efficiency
    max_consumption = user_devices['power'].sum() / 1000 * 16  # 16 hours max usage
    efficiency = max(0, (max_consumption - total_consumption) / max_consumption * 100) if max_consumption > 0 else 0
    
    return {
        "total_consumption": round(total_consumption, 2),
        "monthly_cost": round(monthly_cost, 2),
        "active_devices": active_devices,
        "efficiency": round(efficiency, 1)
    }

# ARIMA prediction endpoint
@app.post("/analytics/predict")
async def predict_consumption(prediction_request: PredictionRequest, current_user: int = Depends(get_current_user)):
    try:
        # Get user's current consumption data
        if devices_df.empty:
            raise HTTPException(status_code=400, detail="No devices found for prediction")
        
        user_devices = devices_df[devices_df['user_id'] == current_user]
        current_consumption = user_devices['consumption'].sum()
        
        # Generate historical data for ARIMA model (simulate 60 days of data)
        historical_data = []
        base_consumption = current_consumption
        
        for i in range(60):
            # Add seasonal and random variations
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * i / 7)  # Weekly seasonality
            trend_factor = 1 + 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.1)
            
            consumption = base_consumption * seasonal_factor * trend_factor * (1 + noise)
            historical_data.append(max(0, consumption))
        
        # Fit ARIMA model
        model = ARIMA(historical_data, order=(2, 1, 2))
        fitted_model = model.fit()
        
        # Make prediction
        forecast = fitted_model.forecast(steps=prediction_request.days)
        predicted_total = forecast.sum()
        
        # Calculate cost
        electricity_rate = 8.0
        predicted_cost = predicted_total * electricity_rate
        
        return {
            "predicted_consumption": round(predicted_total, 2),
            "predicted_cost": round(predicted_cost, 2),
            "daily_average": round(predicted_total / prediction_request.days, 2),
            "prediction_days": prediction_request.days
        }
        
    except Exception as e:
        # Fallback to simple prediction if ARIMA fails
        user_devices = devices_df[devices_df['user_id'] == current_user]
        current_consumption = user_devices['consumption'].sum()
        
        # Simple prediction with seasonal factors
        seasonal_factor = 1 + 0.2 * np.sin(datetime.now().timetuple().tm_yday / 365 * 2 * np.pi)
        predicted_total = current_consumption * prediction_request.days * seasonal_factor
        predicted_cost = predicted_total * 8.0
        
        return {
            "predicted_consumption": round(predicted_total, 2),
            "predicted_cost": round(predicted_cost, 2),
            "daily_average": round(predicted_total / prediction_request.days, 2),
            "prediction_days": prediction_request.days
        }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import hashlib

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_sample_data():
    """Generate sample CSV data files for the energy management system"""
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate sample users
    users_data = [
        {
            'id': 1,
            'name': 'John Doe',
            'email': 'john@example.com',
            'password_hash': hash_password('password123'),
            'avatar': 'J'
        },
        {
            'id': 2,
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'password_hash': hash_password('password123'),
            'avatar': 'J'
        },
        {
            'id': 3,
            'name': 'Mike Johnson',
            'email': 'mike@example.com',
            'password_hash': hash_password('password123'),
            'avatar': 'M'
        }
    ]
    
    users_df = pd.DataFrame(users_data)
    users_df.to_csv('data/users.csv', index=False)
    print("Generated users.csv with sample user data")
    
    # Generate sample devices
    device_types = ['ac', 'heater', 'lights', 'appliance', 'entertainment']
    device_names = {
        'ac': ['Living Room AC', 'Bedroom AC', 'Office AC'],
        'heater': ['Water Heater', 'Room Heater', 'Kitchen Heater'],
        'lights': ['Living Room Lights', 'Bedroom Lights', 'Kitchen Lights', 'Bathroom Lights'],
        'appliance': ['Refrigerator', 'Washing Machine', 'Microwave', 'Dishwasher'],
        'entertainment': ['TV', 'Sound System', 'Gaming Console']
    }
    
    devices_data = []
    device_id = 1
    
    for user_id in [1, 2, 3]:
        # Each user gets 5-8 random devices
        num_devices = np.random.randint(5, 9)
        user_device_types = np.random.choice(device_types, num_devices, replace=True)
        
        for device_type in user_device_types:
            name = np.random.choice(device_names[device_type])
            power_ranges = {
                'ac': (1200, 2000),
                'heater': (1500, 3000),
                'lights': (40, 100),
                'appliance': (100, 800),
                'entertainment': (150, 400)
            }
            
            power = np.random.randint(*power_ranges[device_type])
            status = np.random.choice(['on', 'off'], p=[0.6, 0.4])
            consumption = (power / 1000) * (8 + np.random.random() * 8) if status == 'on' else 0
            
            devices_data.append({
                'id': device_id,
                'user_id': user_id,
                'name': f"{name} - User {user_id}",
                'type': device_type,
                'power': power,
                'status': status,
                'consumption': round(consumption, 2),
                'created_at': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat()
            })
            device_id += 1
    
    devices_df = pd.DataFrame(devices_data)
    devices_df.to_csv('data/devices.csv', index=False)
    print(f"Generated devices.csv with {len(devices_data)} sample devices")
    
    # Generate historical energy data
    energy_data = []
    energy_id = 1
    
    for user_id in [1, 2, 3]:
        user_devices = devices_df[devices_df['user_id'] == user_id]
        
        # Generate 90 days of historical data
        for days_back in range(90):
            date = datetime.now() - timedelta(days=days_back)
            
            for _, device in user_devices.iterrows():
                # Simulate daily consumption with variations
                if device['status'] == 'on':
                    base_consumption = device['consumption']
                    # Add daily variations (±30%)
                    daily_consumption = base_consumption * (0.7 + np.random.random() * 0.6)
                    # Add seasonal effects
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                    daily_consumption *= seasonal_factor
                else:
                    # Sometimes devices are turned on even if currently off
                    daily_consumption = device['power'] / 1000 * np.random.random() * 4 if np.random.random() < 0.3 else 0
                
                if daily_consumption > 0:
                    energy_data.append({
                        'id': energy_id,
                        'user_id': user_id,
                        'device_id': device['id'],
                        'consumption': round(daily_consumption, 3),
                        'timestamp': date.isoformat()
                    })
                    energy_id += 1
    
    energy_df = pd.DataFrame(energy_data)
    energy_df.to_csv('data/energy_data.csv', index=False)
    print(f"Generated energy_data.csv with {len(energy_data)} historical records")
    
    print("\nSample data generation completed!")
    print("Login credentials for testing:")
    print("Email: john@example.com, Password: password123")
    print("Email: jane@example.com, Password: password123")
    print("Email: mike@example.com, Password: password123")

if __name__ == "__main__":
    generate_sample_data()

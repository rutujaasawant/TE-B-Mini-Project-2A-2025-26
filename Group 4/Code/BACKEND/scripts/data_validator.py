import pandas as pd
import numpy as np
from datetime import datetime
import os

def validate_csv_data():
    """Validate the integrity and consistency of CSV data files"""
    
    print("🔍 Validating CSV data files...")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ Data directory not found!")
        return False
    
    validation_results = {}
    
    # Validate users.csv
    try:
        users_df = pd.read_csv('data/users.csv')
        validation_results['users'] = {
            'exists': True,
            'rows': len(users_df),
            'columns': list(users_df.columns),
            'required_columns': ['id', 'name', 'email', 'password_hash', 'avatar'],
            'unique_emails': users_df['email'].nunique() == len(users_df),
            'unique_ids': users_df['id'].nunique() == len(users_df)
        }
        print(f"✅ users.csv: {len(users_df)} users loaded")
    except Exception as e:
        validation_results['users'] = {'exists': False, 'error': str(e)}
        print(f"❌ users.csv: {e}")
    
    # Validate devices.csv
    try:
        devices_df = pd.read_csv('data/devices.csv')
        validation_results['devices'] = {
            'exists': True,
            'rows': len(devices_df),
            'columns': list(devices_df.columns),
            'required_columns': ['id', 'user_id', 'name', 'type', 'power', 'status', 'consumption'],
            'device_types': devices_df['type'].unique().tolist(),
            'status_values': devices_df['status'].unique().tolist(),
            'power_range': [devices_df['power'].min(), devices_df['power'].max()]
        }
        print(f"✅ devices.csv: {len(devices_df)} devices loaded")
        print(f"   Device types: {', '.join(devices_df['type'].unique())}")
    except Exception as e:
        validation_results['devices'] = {'exists': False, 'error': str(e)}
        print(f"❌ devices.csv: {e}")
    
    # Validate energy_data.csv
    try:
        energy_df = pd.read_csv('data/energy_data.csv')
        energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
        validation_results['energy_data'] = {
            'exists': True,
            'rows': len(energy_df),
            'columns': list(energy_df.columns),
            'date_range': [energy_df['timestamp'].min(), energy_df['timestamp'].max()],
            'consumption_range': [energy_df['consumption'].min(), energy_df['consumption'].max()]
        }
        print(f"✅ energy_data.csv: {len(energy_df)} records loaded")
        print(f"   Date range: {energy_df['timestamp'].min()} to {energy_df['timestamp'].max()}")
    except Exception as e:
        validation_results['energy_data'] = {'exists': False, 'error': str(e)}
        print(f"❌ energy_data.csv: {e}")
    
    # Validate device_types.csv
    try:
        device_types_df = pd.read_csv('data/device_types.csv')
        validation_results['device_types'] = {
            'exists': True,
            'rows': len(device_types_df),
            'types': device_types_df['type'].tolist()
        }
        print(f"✅ device_types.csv: {len(device_types_df)} device types loaded")
    except Exception as e:
        validation_results['device_types'] = {'exists': False, 'error': str(e)}
        print(f"❌ device_types.csv: {e}")
    
    # Validate energy_rates.csv
    try:
        rates_df = pd.read_csv('data/energy_rates.csv')
        validation_results['energy_rates'] = {
            'exists': True,
            'rows': len(rates_df),
            'regions': rates_df['region'].tolist(),
            'rate_range': [rates_df['rate_per_kwh'].min(), rates_df['rate_per_kwh'].max()]
        }
        print(f"✅ energy_rates.csv: {len(rates_df)} regions loaded")
    except Exception as e:
        validation_results['energy_rates'] = {'exists': False, 'error': str(e)}
        print(f"❌ energy_rates.csv: {e}")
    
    # Cross-validation checks
    print("\n🔗 Cross-validation checks:")
    
    if validation_results.get('users', {}).get('exists') and validation_results.get('devices', {}).get('exists'):
        try:
            users_df = pd.read_csv('data/users.csv')
            devices_df = pd.read_csv('data/devices.csv')
            
            # Check if all device user_ids exist in users table
            orphaned_devices = devices_df[~devices_df['user_id'].isin(users_df['id'])]
            if len(orphaned_devices) == 0:
                print("✅ All devices have valid user references")
            else:
                print(f"⚠️  {len(orphaned_devices)} devices have invalid user references")
        except Exception as e:
            print(f"❌ Cross-validation error: {e}")
    
    print(f"\n📊 Validation Summary:")
    total_files = len(validation_results)
    valid_files = sum(1 for result in validation_results.values() if result.get('exists', False))
    print(f"   {valid_files}/{total_files} files validated successfully")
    
    return validation_results

def generate_data_statistics():
    """Generate comprehensive statistics about the data"""
    
    print("\n📈 Generating data statistics...")
    
    try:
        users_df = pd.read_csv('data/users.csv')
        devices_df = pd.read_csv('data/devices.csv')
        energy_df = pd.read_csv('data/energy_data.csv')
        
        stats = {
            'total_users': len(users_df),
            'total_devices': len(devices_df),
            'total_energy_records': len(energy_df),
            'devices_per_user': devices_df.groupby('user_id').size().describe().to_dict(),
            'device_type_distribution': devices_df['type'].value_counts().to_dict(),
            'average_power_by_type': devices_df.groupby('type')['power'].mean().to_dict(),
            'total_active_devices': len(devices_df[devices_df['status'] == 'on']),
            'total_consumption': devices_df['consumption'].sum(),
            'average_consumption_per_device': devices_df[devices_df['status'] == 'on']['consumption'].mean()
        }
        
        print("📊 Data Statistics:")
        print(f"   👥 Total Users: {stats['total_users']}")
        print(f"   🔌 Total Devices: {stats['total_devices']}")
        print(f"   📊 Energy Records: {stats['total_energy_records']}")
        print(f"   ⚡ Active Devices: {stats['total_active_devices']}")
        print(f"   🔋 Total Consumption: {stats['total_consumption']:.2f} kWh")
        print(f"   📈 Avg Consumption/Device: {stats['average_consumption_per_device']:.2f} kWh")
        
        print("\n🏠 Device Type Distribution:")
        for device_type, count in stats['device_type_distribution'].items():
            avg_power = stats['average_power_by_type'][device_type]
            print(f"   {device_type}: {count} devices (avg {avg_power:.0f}W)")
        
        return stats
        
    except Exception as e:
        print(f"❌ Statistics generation error: {e}")
        return None

if __name__ == "__main__":
    validation_results = validate_csv_data()
    statistics = generate_data_statistics()
    
    print("\n🎉 Data validation and statistics generation completed!")

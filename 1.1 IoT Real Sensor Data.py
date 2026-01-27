# interactive_data_exploration.ipynb
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load real IoT sensor dataset (Tinybird public dataset)
def load_iot_sensor_data():
    # Option 1: Direct from Tinybird API (real data)
    url = "https://api.us-east.tinybird.co/v0/pipes/iot_sensor_data.json"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()['data']
        df = pd.DataFrame(data)
    except:
        # Option 2: Use sample data if API fails
        print("Using sample IoT data...")
        np.random.seed(42)
        n_samples = 10000
        devices = ['sensor_001', 'sensor_002', 'sensor_003', 'sensor_004', 'sensor_005']
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
            'device_id': np.random.choice(devices, n_samples),
            'temperature': np.random.normal(22, 5, n_samples).round(2),
            'humidity': np.random.normal(45, 15, n_samples).round(2),
            'pressure': np.random.normal(1013, 10, n_samples).round(2),
            'motion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'battery_level': np.random.uniform(20, 100, n_samples).round(2),
            'location': np.random.choice(['factory_a', 'factory_b', 'warehouse'], n_samples),
            'signal_strength': np.random.uniform(0.5, 1.0, n_samples).round(2)
        })
    
    return df

# Interactive exploration
df = load_iot_sensor_data()
print(f"Loaded {len(df)} sensor readings")
print(df.head())

# Interactive visualization
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Temperature Over Time', 'Humidity Distribution', 
                   'Pressure by Location', 'Motion Detection Frequency',
                   'Battery Level Trends', 'Signal Strength Heatmap')
)

# Temperature time series
fig.add_trace(
    go.Scatter(x=df['timestamp'], y=df['temperature'], mode='lines',
               name='Temperature', line=dict(color='red')),
    row=1, col=1
)

# Humidity histogram
fig.add_trace(
    go.Histogram(x=df['humidity'], nbinsx=30, name='Humidity',
                marker_color='blue'),
    row=1, col=2
)

# Pressure by location (box plot)
locations = df['location'].unique()
for location in locations:
    fig.add_trace(
        go.Box(y=df[df['location'] == location]['pressure'],
              name=location),
        row=2, col=1
    )

# Motion detection
motion_counts = df['motion'].value_counts()
fig.add_trace(
    go.Pie(labels=['No Motion', 'Motion'], values=motion_counts.values,
          name='Motion Detection'),
    row=2, col=2
)

# Battery trends by device
for device in df['device_id'].unique()[:3]:  # Show first 3 devices
    device_data = df[df['device_id'] == device]
    fig.add_trace(
        go.Scatter(x=device_data['timestamp'], y=device_data['battery_level'],
                  mode='lines', name=f'Battery {device}'),
        row=3, col=1
    )

# Signal strength heatmap
fig.add_trace(
    go.Densitymapbox(lat=np.random.uniform(40.5, 40.9, len(df)),
                     lon=np.random.uniform(-74.1, -73.9, len(df)),
                     z=df['signal_strength'],
                     radius=10),
    row=3, col=2
)

fig.update_layout(height=1200, showlegend=True, title_text="IoT Sensor Data Dashboard")
fig.show()



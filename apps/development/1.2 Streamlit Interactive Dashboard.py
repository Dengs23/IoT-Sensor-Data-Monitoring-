# app.py - Streamlit dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="IoT Sensor Analytics Dashboard",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 Real-Time IoT Sensor Analytics Dashboard")
st.markdown("Live monitoring of IoT sensor data with anomaly detection")

# Sidebar for controls
st.sidebar.header("Dashboard Controls")
selected_device = st.sidebar.selectbox(
    "Select Device",
    options=['All Devices', 'sensor_001', 'sensor_002', 'sensor_003', 'sensor_004']
)

time_range = st.sidebar.selectbox(
    "Time Range",
    options=['Last 1 hour', 'Last 6 hours', 'Last 24 hours', 'Last 7 days']
)

threshold_temp = st.sidebar.slider(
    "Temperature Alert Threshold (°C)",
    min_value=0, max_value=100, value=30, step=1
)

# Load data (simulated real-time)
@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_data():
    np.random.seed(42)
    n_samples = 1000
    devices = ['sensor_001', 'sensor_002', 'sensor_003', 'sensor_004', 'sensor_005']
    
    df = pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-10', periods=n_samples, freq='1min'),
        'device_id': np.random.choice(devices, n_samples),
        'temperature': np.random.normal(22, 8, n_samples).round(2),
        'humidity': np.random.normal(45, 15, n_samples).round(2),
        'pressure': np.random.normal(1013, 10, n_samples).round(2),
        'motion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'battery_level': np.random.uniform(20, 100, n_samples).round(2),
        'location': np.random.choice(['factory_a', 'factory_b', 'warehouse'], n_samples),
        'signal_strength': np.random.uniform(0.5, 1.0, n_samples).round(2)
    })
    
    # Add some anomalies
    anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[anomaly_indices, 'temperature'] = np.random.uniform(40, 80, 20)
    
    return df

df = load_data()

# Filter data based on selection
if selected_device != 'All Devices':
    df = df[df['device_id'] == selected_device]

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Active Sensors",
        value=df['device_id'].nunique(),
        delta="+2 this week"
    )

with col2:
    avg_temp = df['temperature'].mean()
    st.metric(
        label="Avg Temperature",
        value=f"{avg_temp:.1f}°C",
        delta=f"{(avg_temp - 22):.1f}°C" if avg_temp != 22 else "0°C"
    )

with col3:
    anomalies = len(df[df['temperature'] > threshold_temp])
    st.metric(
        label="Temperature Alerts",
        value=anomalies,
        delta=f"{anomalies} above {threshold_temp}°C",
        delta_color="inverse"
    )

with col4:
    low_battery = len(df[df['battery_level'] < 30])
    st.metric(
        label="Low Battery Sensors",
        value=low_battery,
        delta="⚠️ Needs attention" if low_battery > 0 else "✓ All good"
    )

# Main charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Metrics", "🌡️ Temperature Analysis", 
                                  "📍 Location View", "🔍 Anomaly Detection"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature time series
        fig_temp = px.line(df, x='timestamp', y='temperature', 
                          color='device_id', title='Temperature Over Time')
        fig_temp.add_hline(y=threshold_temp, line_dash="dash", 
                          line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Battery level gauge
        avg_battery = df['battery_level'].mean()
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_battery,
            title={'text': "Avg Battery Level"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "red"},
                    {'range': [20, 40], 'color': "orange"},
                    {'range': [40, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    # Heatmap of temperature by hour and device
    df['hour'] = df['timestamp'].dt.hour
    heatmap_data = df.pivot_table(
        values='temperature', 
        index='device_id', 
        columns='hour', 
        aggfunc='mean'
    )
    
    fig_heat = px.imshow(heatmap_data,
                        labels=dict(x="Hour of Day", y="Device", color="Temperature"),
                        title="Temperature Heatmap by Device and Hour")
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Temperature distribution with histogram
    fig_dist = px.histogram(df, x='temperature', nbins=30,
                           title="Temperature Distribution",
                           marginal="box")
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    # Map visualization (simulated coordinates)
    np.random.seed(42)
    locations = {
        'factory_a': {'lat': 40.7128, 'lon': -74.0060},
        'factory_b': {'lat': 40.7589, 'lon': -73.9851},
        'warehouse': {'lat': 40.7549, 'lon': -73.9840}
    }
    
    # Create map data
    map_data = []
    for loc in df['location'].unique():
        loc_df = df[df['location'] == loc]
        map_data.append({
            'location': loc,
            'lat': locations[loc]['lat'] + np.random.uniform(-0.01, 0.01, len(loc_df)),
            'lon': locations[loc]['lon'] + np.random.uniform(-0.01, 0.01, len(loc_df)),
            'temperature': loc_df['temperature'].values,
            'device_count': len(loc_df['device_id'].unique())
        })
    
    map_df = pd.DataFrame(map_data)
    
    # Create bubble map
    fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon",
                               size="device_count",
                               color="temperature",
                               size_max=30,
                               zoom=11,
                               mapbox_style="carto-positron",
                               title="Sensor Locations & Temperature")
    st.plotly_chart(fig_map, use_container_width=True)

with tab4:
    # Anomaly detection interface
    st.header("Anomaly Detection Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configure Detection Parameters")
        
        method = st.selectbox(
            "Detection Method",
            ["Statistical Threshold", "Machine Learning", "Moving Average"]
        )
        
        if method == "Statistical Threshold":
            z_score = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.5)
            st.info(f"Values beyond {z_score} standard deviations will be flagged")
        
        elif method == "Machine Learning":
            model_type = st.selectbox("Model Type", ["Isolation Forest", "One-Class SVM", "Autoencoder"])
            contamination = st.slider("Expected Anomaly Ratio", 0.01, 0.2, 0.05, 0.01)
        
        st.button("Train/Update Model", type="primary")
    
    with col2:
        st.subheader("Live Anomaly Detection")
        
        # Simulate real-time anomalies
        placeholder = st.empty()
        
        for seconds in range(30):
            with placeholder.container():
                # Generate fake real-time data
                current_time = datetime.now()
                fake_data = {
                    'timestamp': current_time,
                    'device_id': np.random.choice(['sensor_001', 'sensor_002']),
                    'temperature': np.random.normal(22, 10),
                    'is_anomaly': np.random.random() > 0.8
                }
                
                # Display
                st.metric("Current Reading", f"{fake_data['temperature']:.1f}°C")
                
                if fake_data['is_anomaly']:
                    st.error(f"🚨 ANOMALY DETECTED on {fake_data['device_id']}")
                    st.write(f"Temperature: {fake_data['temperature']:.1f}°C at {current_time.strftime('%H:%M:%S')}")
                else:
                    st.success("✓ Normal operation")
                
                # Add to historical alerts
                if fake_data['is_anomaly']:
                    st.session_state.setdefault('alerts', []).append(fake_data)
            
            time.sleep(1)  # Simulate 1-second updates
        
        # Show recent alerts
        if 'alerts' in st.session_state and st.session_state['alerts']:
            st.subheader("Recent Alerts")
            alerts_df = pd.DataFrame(st.session_state['alerts'])
            st.dataframe(alerts_df)

# Real-time data stream simulation
st.sidebar.header("Simulate Data Stream")
if st.sidebar.button("Start Data Stream"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        # Update progress
        progress_bar.progress(i + 1)
        status_text.text(f"Processing data point {i + 1}/100")
        
        # Simulate data processing
        time.sleep(0.05)
    
    st.sidebar.success("✅ Data stream simulation complete!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    **IoT Sensor Analytics Platform**
    - Real-time monitoring
    - Anomaly detection
    - Predictive maintenance
    - Built with Streamlit + Azure
    """
)



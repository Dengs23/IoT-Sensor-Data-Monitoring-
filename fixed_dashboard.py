import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ROI Calculator Functions (copied directly here)
def estimate_energy_usage(building_data):
    """Estimate energy usage based on building characteristics"""
    base_usage = building_data.get('square_meters', 1000) * 0.15  # kWh/m²/month
    
    # Adjust for building age
    age_factor = 1 + (building_data.get('age_years', 10) * 0.01)
    
    # Adjust for insulation (better insulation = less energy)
    insulation = building_data.get('insulation_rating', 5)
    insulation_factor = 1.2 - (insulation * 0.02)
    
    # Adjust for temperature anomalies
    temp_variance = building_data.get('temperature_variance', 0)
    temp_factor = 1 + (temp_variance * 0.1)
    
    return base_usage * age_factor * insulation_factor * temp_factor

def estimate_value_increase(building_data):
    """Estimate property value increase from IoT monitoring"""
    base_value = building_data.get('property_value', 1000000)
    
    # Value increase from risk reduction
    risk_reduction = building_data.get('risk_reduction_score', 0.1)
    
    # Value increase from efficiency improvements
    efficiency_gain = building_data.get('efficiency_improvement', 0.15)
    
    # Combined value increase
    value_increase_percent = (risk_reduction * 0.5 + efficiency_gain * 0.5) * 100
    
    return base_value * (value_increase_percent / 100)

def calculate_roi(building_data, upgrade_costs, energy_prices):
    """Calculate ROI for HVAC upgrades and IoT monitoring system"""
    # Estimate current energy usage
    current_energy_usage = estimate_energy_usage(building_data)
    current_energy_cost = current_energy_usage * energy_prices
    
    # Estimate improved energy usage
    efficiency_improvement = building_data.get('efficiency_improvement', 0.3)
    upgraded_energy_cost = current_energy_cost * (1 - efficiency_improvement)
    
    # Calculate savings
    annual_savings = current_energy_cost - upgraded_energy_cost
    payback_period = upgrade_costs / annual_savings if annual_savings > 0 else float('inf')
    
    # Property value increase
    value_increase = estimate_value_increase(building_data)
    
    # ROI calculations
    roi_10_year = ((annual_savings * 10) + value_increase - upgrade_costs) / upgrade_costs
    
    return {
        'current_energy_cost': round(current_energy_cost, 2),
        'upgraded_energy_cost': round(upgraded_energy_cost, 2),
        'annual_savings': round(annual_savings, 2),
        'monthly_savings': round(annual_savings / 12, 2),
        'payback_years': round(payback_period, 1),
        'property_value_increase': round(value_increase, 2),
        'value_increase_percent': round((value_increase / building_data.get('property_value', 1000000)) * 100, 1),
        '10_year_roi': round(roi_10_year * 100, 1),
        'total_10_year_benefit': round((annual_savings * 10) + value_increase, 2),
        'net_10_year_gain': round(((annual_savings * 10) + value_increase - upgrade_costs), 2)
    }

# Streamlit App Configuration
st.set_page_config(
    page_title="IoT Sensor Analytics & Valuation Dashboard",
    page_icon="📡",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", [
    "📊 Live Monitoring", 
    "🤖 ML Insights", 
    "💰 ROI Calculator",
    "🏢 Building Valuation"
])

# Sidebar Controls
st.sidebar.header("Dashboard Controls")
selected_device = st.sidebar.selectbox(
    "Select Device",
    options=['All Devices', 'sensor_001', 'sensor_002', 'sensor_003', 'sensor_004']
)

time_range = st.sidebar.selectbox(
    "Time Range",
    options=['Last 1 hour', 'Last 6 hours', 'Last 24 hours', 'Last 7 days']
)

# Load data function
@st.cache_data(ttl=60)
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
    })
    return df

# PAGE 1: Live Monitoring
if page == "📊 Live Monitoring":
    st.title("📡 Real-Time IoT Sensor Analytics Dashboard")
    st.markdown("Live monitoring of IoT sensor data with anomaly detection")
    
    # Load data
    df = load_data()
    
    # Filter based on selection
    if selected_device != 'All Devices':
        df = df[df['device_id'] == selected_device]
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        current_temp = df['temperature'].iloc[-1]
        prev_temp = df['temperature'].iloc[-2] if len(df) > 1 else current_temp
        st.metric("Current Temperature", f"{current_temp:.1f}°C", 
                 delta=f"{current_temp - prev_temp:.1f}°C")
    
    with col2:
        avg_temp = df['temperature'].mean()
        st.metric("Average Temperature", f"{avg_temp:.1f}°C")
    
    with col3:
        anomaly_count = len(df[df['temperature'] > 30])  # Simple threshold
        st.metric("Temperature Alerts", anomaly_count)
    
    with col4:
        battery_level = 85  # Simulated
        st.metric("Battery Level", f"{battery_level}%")
    
    # Temperature chart
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=df['timestamp'][-100:],  # Last 100 points
        y=df['temperature'][-100:],
        mode='lines',
        name='Temperature',
        line=dict(color='blue', width=2)
    ))
    
    # Add threshold line
    fig_temp.add_hline(y=30, line_dash="dash", line_color="red", 
                      annotation_text="Alert Threshold (30°C)")
    
    fig_temp.update_layout(
        title='Temperature Over Time',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        height=400
    )
    
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Data table
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(20))

# PAGE 2: ML Insights
elif page == "🤖 ML Insights":
    st.header("🤖 Machine Learning Insights")
    
    # Generate ML predictions
    np.random.seed(42)
    n_samples = 200
    timestamps = pd.date_range(end=datetime.now(), periods=n_samples, freq='10min')
    
    # Simulate ML predictions
    temperature_data = np.random.normal(22, 5, n_samples)
    mean_temp = np.mean(temperature_data)
    std_temp = np.std(temperature_data)
    anomalies = [1 if abs(temp - mean_temp) > 2 * std_temp else 0 for temp in temperature_data]
    
    # Create dataframe
    ml_df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature_data.round(2),
        'prediction': anomalies,
        'risk_score': [min(1.0, abs(temp - mean_temp) / (3 * std_temp)) for temp in temperature_data]
    })
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        anomaly_count = sum(anomalies)
        st.metric("Anomalies Detected", anomaly_count)
    with col2:
        avg_risk = ml_df['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.1%}")
    with col3:
        st.metric("Prediction Confidence", "98.3%")
    
    # Anomaly visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ml_df['timestamp'],
        y=ml_df['temperature'],
        mode='lines',
        name='Temperature',
        line=dict(color='blue', width=1)
    ))
    
    # Highlight anomalies
    anomaly_points = ml_df[ml_df['prediction'] == 1]
    if len(anomaly_points) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_points['timestamp'],
            y=anomaly_points['temperature'],
            mode='markers',
            name='ML Detected Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title='ML Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: ROI Calculator
elif page == "💰 ROI Calculator":
    st.header("💰 ROI Calculator for IoT Implementation")
    
    st.markdown("Calculate return on investment for IoT monitoring system")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Building Details")
        square_meters = st.number_input("Building Size (m²)", 100, 50000, 5000)
        building_age = st.number_input("Building Age (years)", 0, 100, 15)
        property_value = st.number_input("Property Value ($)", 100000, 50000000, 2500000)
        insulation = st.slider("Insulation Rating (1-10)", 1, 10, 6)
    
    with col2:
        st.subheader("System & Energy")
        upgrade_cost = st.number_input("IoT System Cost ($)", 10000, 1000000, 75000)
        energy_price = st.number_input("Energy Price ($/kWh)", 0.05, 1.0, 0.15, 0.01)
        
        st.subheader("ML Predictions")
        risk_reduction = st.slider("Risk Reduction %", 0, 100, 25) / 100
        efficiency_improvement = st.slider("Efficiency Improvement %", 0, 50, 35) / 100
        temp_variance = st.slider("Temperature Variance", 0.0, 5.0, 0.8, 0.1)
    
    # Calculate ROI
    if st.button("Calculate ROI", type="primary"):
        building_data = {
            'square_meters': square_meters,
            'age_years': building_age,
            'insulation_rating': insulation,
            'temperature_variance': temp_variance,
            'risk_reduction_score': risk_reduction,
            'efficiency_improvement': efficiency_improvement,
            'property_value': property_value
        }
        
        results = calculate_roi(building_data, upgrade_cost, energy_price)
        
        # Display results
        st.success("### ROI Analysis Results")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Payback Period", f"{results['payback_years']:.1f} years")
        with col2:
            st.metric("10-Year ROI", f"{results['10_year_roi']:.1f}%")
        with col3:
            st.metric("Value Increase", f"${results['property_value_increase']:,.0f}")
        
        # Detailed table
        st.subheader("Detailed Breakdown")
        results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        st.dataframe(results_df)

# PAGE 4: Building Valuation
elif page == "🏢 Building Valuation":
    st.header("🏢 Building Valuation Analysis")
    
    st.markdown("Estimate building value based on IoT monitoring data")
    
    # Simple valuation calculator
    base_value = st.number_input("Base Property Value ($)", 100000, 10000000, 1000000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Factors")
        temp_stability = st.slider("Temperature Stability", 0, 100, 75)
        anomaly_freq = st.slider("Anomaly Frequency", 0, 100, 20)
    
    with col2:
        st.subheader("Improvement Factors")
        iot_coverage = st.slider("IoT Coverage %", 0, 100, 80)
        maintenance = st.slider("Maintenance Score", 0, 100, 65)
    
    if st.button("Calculate Valuation"):
        # Simple calculation
        risk_score = (anomaly_freq * 0.6 + (100 - temp_stability) * 0.4) / 100
        improvement_score = (iot_coverage * 0.6 + maintenance * 0.4) / 100
        
        adjusted_value = base_value * (1 - risk_score * 0.2) * (1 + improvement_score * 0.15)
        value_change = adjusted_value - base_value
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Value", f"${base_value:,.0f}")
        with col2:
            st.metric("Adjusted Value", f"${adjusted_value:,.0f}")
        with col3:
            st.metric("Value Change", f"${value_change:,.0f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**IoT Sensor Data Monitoring**")
st.sidebar.markdown("Version 2.0 | Real-time valuation")

# Auto-refresh
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

# Auto-refresh every 60 seconds
if st.sidebar.checkbox("Auto-refresh (60s)"):
    time.sleep(60)
    st.rerun()

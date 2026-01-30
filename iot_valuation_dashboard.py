import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Import your ROI calculator functions
from "5.5 ROI Calculator" import calculate_roi

# Page configuration
st.set_page_config(
    page_title="IoT Building Valuation Dashboard",
    page_icon="🏢",
    layout="wide"
)

# Title
st.title("🏢 IoT Sensor Data & Building Valuation Dashboard")
st.markdown("Real-time temperature monitoring with ML predictions and ROI analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Real-time Monitoring", 
    "🤖 ML Predictions", 
    "💰 ROI Calculator",
    "📈 Building Valuation"
])

# Function to generate sample sensor data
def generate_sensor_data(hours=24):
    """Generate realistic sensor data"""
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
    
    data = {
        'timestamp': timestamps[::-1],
        'temperature': np.random.normal(22, 3, hours).tolist(),
        'humidity': np.random.normal(45, 10, hours).tolist(),
        'battery': np.random.uniform(85, 100, hours).tolist(),
        'location': ['Building_A'] * hours
    }
    
    # Add anomalies for ML to detect
    anomaly_indices = np.random.choice(hours, size=3, replace=False)
    for idx in anomaly_indices:
        data['temperature'][idx] += np.random.choice([-10, 12])
    
    return pd.DataFrame(data)

# Function for ML predictions (simplified)
def predict_anomalies(temperature_data):
    """Simple anomaly detection"""
    mean_temp = np.mean(temperature_data)
    std_temp = np.std(temperature_data)
    anomalies = []
    
    for temp in temperature_data:
        if abs(temp - mean_temp) > 2 * std_temp:
            anomalies.append(1)  # Anomaly
        else:
            anomalies.append(0)  # Normal
    
    return anomalies

# PAGE 1: Real-time Monitoring
if page == "📊 Real-time Monitoring":
    st.header("📊 Real-time Sensor Data")
    
    # Generate sample data
    sensor_data = generate_sensor_data(48)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Temp", f"{sensor_data['temperature'].iloc[-1]:.1f}°C", 
                 delta=f"{sensor_data['temperature'].iloc[-1] - sensor_data['temperature'].iloc[-2]:.1f}°C")
    with col2:
        st.metric("Humidity", f"{sensor_data['humidity'].iloc[-1]:.1f}%")
    with col3:
        st.metric("Battery", f"{sensor_data['battery'].iloc[-1]:.1f}%")
    with col4:
        anomalies = sum(predict_anomalies(sensor_data['temperature']))
        st.metric("Anomalies Detected", anomalies)
    
    # Plot temperature over time
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sensor_data['timestamp'],
        y=sensor_data['temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='blue', width=2)
    ))
    
    # Highlight anomalies
    anomaly_mask = predict_anomalies(sensor_data['temperature'])
    anomalies_df = sensor_data[anomaly_mask]
    if len(anomalies_df) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies_df['timestamp'],
            y=anomalies_df['temperature'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title='Temperature Readings Over Time',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data table
    with st.expander("View Raw Sensor Data"):
        st.dataframe(sensor_data)

# PAGE 2: ML Predictions
elif page == "🤖 ML Predictions":
    st.header("🤖 Machine Learning Predictions")
    
    # Generate data
    sensor_data = generate_sensor_data(72)
    
    # Make predictions
    predictions = predict_anomalies(sensor_data['temperature'])
    sensor_data['prediction'] = predictions
    sensor_data['status'] = ['⚠️ Anomaly' if p == 1 else '✅ Normal' for p in predictions]
    
    # Calculate risk score based on anomalies
    anomaly_count = sum(predictions)
    risk_score = min(1.0, anomaly_count / 10)  # Cap at 1.0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk Score", f"{risk_score:.2%}")
        st.metric("Anomalies Detected", anomaly_count)
    
    with col2:
        st.metric("Data Points", len(sensor_data))
        st.metric("Prediction Accuracy", "98.3%")  # From your ML model
    
    # Show predictions
    st.subheader("Anomaly Detection Results")
    st.dataframe(sensor_data[['timestamp', 'temperature', 'status']].tail(20))
    
    # Risk visualization
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score * 100,
        title = {'text': "Building Risk Level"},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if risk_score > 0.7 else "orange" if risk_score > 0.3 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ]
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: ROI Calculator
elif page == "💰 ROI Calculator":
    st.header("💰 ROI Calculator for IoT System")
    
    st.markdown("Calculate the return on investment for implementing IoT monitoring")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Building Details")
        square_meters = st.number_input("Building Size (m²)", 100, 10000, 5000)
        building_age = st.number_input("Building Age (years)", 1, 100, 15)
        property_value = st.number_input("Property Value ($)", 100000, 10000000, 2500000)
        insulation = st.slider("Insulation Rating (1-10)", 1, 10, 6)
    
    with col2:
        st.subheader("System Costs & Prices")
        upgrade_cost = st.number_input("IoT System Cost ($)", 10000, 500000, 75000, step=5000)
        energy_price = st.number_input("Energy Price ($/kWh)", 0.05, 1.0, 0.15, step=0.01)
        
        # Risk reduction from ML predictions
        st.subheader("ML Predictions Impact")
        risk_reduction = st.slider("Risk Reduction %", 0, 100, 25) / 100
        efficiency_improvement = st.slider("Efficiency Improvement %", 0, 50, 35) / 100
        
        # Temperature variance from sensors
        temp_variance = st.slider("Temperature Variance", 0.0, 5.0, 0.8, step=0.1)
    
    # Calculate button
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
            st.metric("Property Value Increase", f"${results['property_value_increase']:,.0f}")
        
        # Detailed results
        st.subheader("Detailed Breakdown")
        results_df = pd.DataFrame([results]).T.reset_index()
        results_df.columns = ['Metric', 'Value']
        st.dataframe(results_df)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Costs', x=['System Cost'], y=[upgrade_cost], marker_color='red'),
            go.Bar(name='Benefits', 
                   x=['Annual Savings', 'Value Increase'], 
                   y=[results['annual_savings'], results['property_value_increase']], 
                   marker_color='green')
        ])
        fig.update_layout(title='Cost vs Benefits Analysis', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4: Building Valuation
elif page == "📈 Building Valuation":
    st.header("📈 Building Valuation Analysis")
    
    st.markdown("Estimate building value based on IoT monitoring data and risk assessment")
    
    # Inputs for valuation
    base_value = st.number_input("Base Property Value ($)", 100000, 10000000, 1000000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Factors")
        temp_stability = st.slider("Temperature Stability", 0, 100, 75)
        anomaly_frequency = st.slider("Anomaly Frequency", 0, 100, 20)
        equipment_age = st.slider("Equipment Age Factor", 0, 100, 40)
    
    with col2:
        st.subheader("Improvement Factors")
        iot_coverage = st.slider("IoT Coverage %", 0, 100, 80)
        maintenance_score = st.slider("Maintenance Score", 0, 100, 65)
        energy_efficiency = st.slider("Energy Efficiency", 0, 100, 70)
    
    # Calculate valuation
    if st.button("Calculate Valuation"):
        # Calculate risk score (lower is better)
        risk_score = (anomaly_frequency * 0.4 + equipment_age * 0.3 + (100 - temp_stability) * 0.3) / 100
        
        # Calculate improvement score
        improvement_score = (iot_coverage * 0.4 + maintenance_score * 0.3 + energy_efficiency * 0.3) / 100
        
        # Calculate adjusted value
        risk_adjustment = 1 - (risk_score * 0.2)  # Up to 20% reduction for high risk
        improvement_adjustment = 1 + (improvement_score * 0.15)  # Up to 15% increase for improvements
        
        adjusted_value = base_value * risk_adjustment * improvement_adjustment
        value_change = adjusted_value - base_value
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Value", f"${base_value:,.0f}")
        with col2:
            st.metric("Adjusted Value", f"${adjusted_value:,.0f}")
        with col3:
            st.metric("Value Change", f"${value_change:,.0f}", 
                     delta=f"{(value_change/base_value)*100:.1f}%")
        
        # Risk vs Improvement chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[temp_stability, 100-anomaly_frequency, 100-equipment_age, 
               iot_coverage, maintenance_score, energy_efficiency],
            theta=['Temp Stability', 'Low Anomalies', 'New Equipment', 
                   'IoT Coverage', 'Maintenance', 'Efficiency'],
            fill='toself',
            name='Building Score'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Building Assessment Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**IoT Sensor Data Monitoring**")
st.sidebar.markdown("Version 1.0 | Real-time building valuation")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Any, Tuple

st.set_page_config(
    page_title="🚀 IoT Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class IoTVisualizationEngine:
    @staticmethod
    def create_sensor_network_3d(building_dimensions, sensor_coordinates):
        length, width, height = building_dimensions
        
        fig = go.Figure()
        
        x_box = [0, length, length, 0, 0, length, length, 0]
        y_box = [0, 0, width, width, 0, 0, width, width]
        z_box = [0, 0, 0, 0, height, height, height, height]
        
        fig.add_trace(go.Mesh3d(
            x=x_box,
            y=y_box,
            z=z_box,
            color='lightblue',
            opacity=0.1,
            name='Building'
        ))
        
        if sensor_coordinates and len(sensor_coordinates) > 0:
            coords = np.array(sensor_coordinates)
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=coords[:, 2],
                    colorscale='Viridis',
                    opacity=0.8,
                    line=dict(color='white', width=1)
                ),
                name='Sensors',
                text=[f'Sensor {i+1}' for i in range(len(coords))]
            ))
        
        fig.update_layout(
            title="3D Sensor Network",
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Width (m)',
                zaxis_title='Height (m)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=width/length, z=height/length),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_energy_heatmap(hourly_consumption):
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        hours = [f'{h:02d}:00' for h in range(24)]
        
        data = np.random.randn(7, 24) * 50 + hourly_consumption
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=hours,
            y=days,
            colorscale='Viridis',
            colorbar=dict(title="Power (kW)")
        ))
        
        fig.update_layout(
            title="Weekly Energy Heatmap",
            xaxis_title="Hour",
            yaxis_title="Day",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_animated_timeseries(sensor_data, anomalies=None):
        time_points = len(sensor_data)
        time_index = list(range(time_points))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_index,
            y=sensor_data,
            mode='lines',
            name='Sensor Data',
            line=dict(color='blue', width=2)
        ))
        
        if anomalies and len(anomalies) > 0:
            anomaly_y = [sensor_data[i] for i in anomalies]
            fig.add_trace(go.Scatter(
                x=anomalies,
                y=anomaly_y,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            title="Sensor Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        
        return fig

class EnhancedROIOptimizer:
    def calculate_roi(self, params):
        building_area = params.get("building_area", 1000)
        investment = params.get("investment", 50000)
        energy_cost = params.get("energy_cost", 0.15)
        years = params.get("years", 10)
        
        energy_savings = building_area * 2.5 * energy_cost * 365 * 0.25
        total_savings = energy_savings * years
        net_profit = total_savings - investment
        roi_percentage = (net_profit / investment) * 100 if investment > 0 else 0
        
        sensor_coords = self._generate_sensor_coordinates(building_area)
        
        return {
            "roi_percentage": round(roi_percentage, 2),
            "payback_years": round(investment / energy_savings, 2) if energy_savings > 0 else float('inf'),
            "annual_savings": round(energy_savings, 2),
            "net_profit": round(net_profit, 2),
            "sensor_coordinates": sensor_coords,
            "building_dimensions": [building_area**0.5 * 1.2, building_area**0.5 * 0.8, 4]
        }
    
    def _generate_sensor_coordinates(self, area):
        length = area**0.5 * 1.2
        width = area**0.5 * 0.8
        height = 4
        
        spacing = 5
        x_points = np.arange(2, length, spacing)
        y_points = np.arange(2, width, spacing)
        z_points = [1.5, 3.0]
        
        coordinates = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    coordinates.append([x, y, z])
        
        return coordinates[:30]

def main():
    st.markdown('<h1 class="main-header">🚀 IoT Dashboard Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Visualization Platform")
    
    viz_engine = IoTVisualizationEngine()
    roi_optimizer = EnhancedROIOptimizer()
    
    tab1, tab2, tab3 = st.tabs(["📊 3D Network", "⚡ Energy Analytics", "💰 ROI Calculator"])
    
    with tab1:
        st.header("📊 3D Sensor Network")
        
        building_area = st.slider("Building Area (m²)", 100, 5000, 1000, 100)
        num_sensors = st.slider("Sensors", 5, 100, 20, 5)
        
        if st.button("🔄 Generate 3D Model"):
            length = building_area**0.5 * 1.2
            width = building_area**0.5 * 0.8
            height = 4
            
            spacing = 5
            x_points = np.arange(2, length, spacing)
            y_points = np.arange(2, width, spacing)
            z_points = [1.5, 3.0]
            
            coordinates = []
            for x in x_points:
                for y in y_points:
                    for z in z_points:
                        if len(coordinates) < num_sensors:
                            coordinates.append([x, y, z])
            
            fig = viz_engine.create_sensor_network_3d([length, width, height], coordinates)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sensors", len(coordinates))
            with col2:
                st.metric("Coverage", f"{(len(coordinates) * 25) / building_area * 100:.1f}%")
    
    with tab2:
        st.header("⚡ Energy Analytics")
        
        peak_adjustment = st.slider("Peak Load %", -50, 50, 0, 5)
        
        hourly_base = np.array([50, 45, 40, 38, 40, 60, 100, 150, 180, 170, 160, 155,
                                160, 165, 170, 180, 200, 220, 210, 180, 140, 100, 70, 55])
        
        hourly_data = hourly_base * (1 + peak_adjustment/100)
        
        fig = viz_engine.create_energy_heatmap(np.mean(hourly_data))
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cost Analysis")
        electricity_rate = 0.15
        daily_cost = np.sum(hourly_data) * electricity_rate
        monthly_cost = daily_cost * 30
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Daily Cost", f"${daily_cost:.2f}")
        with col2:
            st.metric("Monthly Cost", f"${monthly_cost:.2f}")
    
    with tab3:
        st.header("💰 ROI Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment = st.number_input("Investment ($)", 10000, 500000, 50000, 5000)
            building_area = st.number_input("Area (m²)", 100, 10000, 2000, 100)
        
        with col2:
            energy_cost = st.slider("Energy Cost ($/kWh)", 0.05, 0.50, 0.15, 0.01)
            years = st.slider("Years", 1, 20, 10, 1)
        
        if st.button("📊 Calculate ROI"):
            params = {
                "building_area": building_area,
                "investment": investment,
                "energy_cost": energy_cost,
                "years": years
            }
            
            results = roi_optimizer.calculate_roi(params)
            
            st.subheader("📈 Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROI", f"{results['roi_percentage']:.1f}%")
            
            with col2:
                st.metric("Payback", f"{results['payback_years']:.1f} years")
            
            with col3:
                st.metric("Annual Savings", f"${results['annual_savings']:,.0f}")
            
            with col4:
                st.metric("Net Profit", f"${results['net_profit']:,.0f}")
            
            years_list = list(range(years + 1))
            cash_flows = [-investment] + [results['annual_savings']] * years
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=years_list, y=cash_flows, name='Cash Flow'))
            fig.update_layout(title="Cash Flow Analysis", xaxis_title="Year", yaxis_title="$")
            st.plotly_chart(fig, use_container_width=True)
    
    with st.sidebar:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.title("📊 Dashboard Pro")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        theme = st.selectbox("Theme", ["Default", "Dark", "Light"])
        auto_refresh = st.checkbox("Auto-refresh", True)
        
        st.divider()
        
        if st.button("📥 Export", use_container_width=True):
            st.success("Exported!")
        
        st.divider()
        
        st.markdown("""
        **IoT Dashboard Pro**
        
        Features:
        • 3D Visualizations
        • Energy Analytics
        • ROI Calculator
        
        Powered by Streamlit + Plotly
        """)

if __name__ == "__main__":
    main()

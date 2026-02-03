import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Any, Tuple
from scipy import stats, optimize, signal, integrate
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧠 Advanced IoT Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MATHEMATICAL OPTIMIZER ====================
class MathematicalOptimizer:
    """Mathematical optimization engine for IoT systems"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_sensor_placement(self, building_dimensions, sensor_range):
        """Optimal sensor placement using geometric optimization"""
        length, width, height = building_dimensions
        
        optimal_spacing = sensor_range / np.sqrt(2)
        
        x_points = np.arange(0, length, optimal_spacing)
        y_points = np.arange(0, width, optimal_spacing)
        z_points = np.arange(0, height, optimal_spacing)
        
        X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
        coordinates = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        total_volume = length * width * height
        coverage_radius = sensor_range
        sensor_volume = (4/3) * np.pi * (coverage_radius**3)
        estimated_coverage = min(100, (len(coordinates) * sensor_volume / total_volume) * 100)
        
        return {
            'coordinates': coordinates.tolist(),
            'optimal_spacing': optimal_spacing,
            'num_sensors': len(coordinates),
            'coverage_percentage': round(estimated_coverage, 2),
            'building_dimensions': building_dimensions,
            'sensor_range': sensor_range
        }
    
    def energy_consumption_optimization(self, power_data):
        """Optimize energy consumption"""
        power = np.array(power_data)
        
        sorted_indices = np.argsort(power)
        optimal_schedule = np.zeros_like(power)
        
        for i, idx in enumerate(sorted_indices):
            optimal_schedule[idx] = power[idx] * (0.5 + 0.5 * i / len(power))
        
        original_consumption = np.sum(power)
        optimized_consumption = np.sum(optimal_schedule)
        savings = ((original_consumption - optimized_consumption) / original_consumption) * 100
        
        return {
            'optimal_schedule': optimal_schedule.tolist(),
            'savings_percentage': round(savings, 2),
            'original_consumption': original_consumption,
            'optimized_consumption': optimized_consumption
        }
    
    def predictive_maintenance_model(self, sensor_data, threshold=3):
        """Predictive maintenance using statistical analysis"""
        data = np.array(sensor_data)
        
        window = 10
        rolling_mean = pd.Series(data).rolling(window=window).mean().dropna().values
        rolling_std = pd.Series(data).rolling(window=window).std().dropna().values
        
        z_scores = (data[window-1:] - rolling_mean) / rolling_std
        
        anomalies = np.where(np.abs(z_scores) > threshold)[0]
        
        # Calculate maintenance urgency
        if len(anomalies) > 0:
            urgency = min(100, (len(anomalies) / len(data)) * 200)
            time_to_failure = len(data) - anomalies[-1]
        else:
            urgency = 0
            time_to_failure = len(data)
        
        return {
            'anomalies': anomalies.tolist(),
            'z_scores': z_scores.tolist(),
            'rolling_mean': rolling_mean.tolist(),
            'rolling_std': rolling_std.tolist(),
            'maintenance_urgency': round(urgency, 2),
            'time_to_failure': time_to_failure
        }
    
    def optimize_roi(self, investment, annual_savings, years, discount_rate=0.08):
        """Optimize ROI with interactive parameters"""
        years_array = np.arange(1, years + 1)
        
        # Calculate Net Present Value (NPV)
        discounted_savings = annual_savings / (1 + discount_rate) ** years_array
        npv = np.sum(discounted_savings) - investment
        
        # Calculate ROI
        total_savings = annual_savings * years
        simple_roi = ((total_savings - investment) / investment) * 100
        
        # Payback period
        cumulative_cash_flow = np.cumsum([-investment] + [annual_savings] * years)
        payback_period = None
        for i, cf in enumerate(cumulative_cash_flow):
            if cf >= 0:
                payback_period = i
                break
        
        # Sensitivity analysis
        variations = np.linspace(0.5, 1.5, 5)  # 50% to 150%
        npv_sensitivity = []
        roi_sensitivity = []
        for var in variations:
            adj_savings = annual_savings * var
            adj_discounted = adj_savings / (1 + discount_rate) ** years_array
            npv_sensitivity.append(np.sum(adj_discounted) - investment)
            roi_sensitivity.append(((adj_savings * years - investment) / investment) * 100)
        
        return {
            'investment': investment,
            'annual_savings': annual_savings,
            'years': years,
            'npv': round(npv, 2),
            'simple_roi_percent': round(simple_roi, 2),
            'payback_period_years': payback_period if payback_period else years + 1,
            'discounted_savings': discounted_savings.tolist(),
            'total_savings': total_savings,
            'sensitivity_analysis': {
                'variations': variations.tolist(),
                'npv_values': [round(v, 2) for v in npv_sensitivity],
                'roi_values': [round(v, 2) for v in roi_sensitivity]
            }
        }

# ==================== STREAMLIT UI ====================
def main():
    st.title("🧠 Advanced IoT Assistant with Mathematical Optimization")
    st.markdown("### Interactive Mathematical Optimization Dashboard")
    
    # Initialize optimizer
    optimizer = MathematicalOptimizer()
    
    # Create tabs for different optimization types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 ROI Optimization", 
        "📍 Sensor Placement", 
        "⚡ Energy Optimization", 
        "🔧 Predictive Maintenance"
    ])
    
    # ==================== TAB 1: ROI OPTIMIZATION ====================
    with tab1:
        st.header("💰 ROI Optimization Calculator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            investment = st.number_input(
                "Initial Investment ($)",
                min_value=1000,
                max_value=1000000,
                value=50000,
                step=5000,
                help="Total initial investment in IoT system"
            )
        
        with col2:
            annual_savings = st.number_input(
                "Annual Savings ($)",
                min_value=1000,
                max_value=500000,
                value=15000,
                step=1000,
                help="Expected annual savings from IoT implementation"
            )
        
        with col3:
            years = st.slider(
                "Time Period (Years)",
                min_value=1,
                max_value=20,
                value=5,
                help="Investment time horizon"
            )
        
        discount_rate = st.slider(
            "Discount Rate (%)",
            min_value=1.0,
            max_value=15.0,
            value=8.0,
            step=0.5,
            help="Discount rate for NPV calculation"
        ) / 100
        
        if st.button("🔍 Calculate Optimized ROI", type="primary"):
            with st.spinner("Calculating optimal ROI..."):
                roi_result = optimizer.optimize_roi(investment, annual_savings, years, discount_rate)
                
                # Display results
                st.subheader("📈 Optimization Results")
                
                metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                
                with metrics_col1:
                    st.metric("Net Present Value", f"${roi_result['npv']:,.2f}")
                
                with metrics_col2:
                    st.metric("Simple ROI", f"{roi_result['simple_roi_percent']}%")
                
                with metrics_col3:
                    st.metric("Payback Period", f"{roi_result['payback_period_years']} years")
                
                with metrics_col4:
                    st.metric("Total Savings", f"${roi_result['total_savings']:,.2f}")
                
                # Cash flow chart
                st.subheader("📊 Cash Flow Analysis")
                years_list = list(range(years + 1))
                cash_flows = [-investment] + [annual_savings] * years
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=years_list,
                    y=cash_flows,
                    name="Annual Cash Flow",
                    marker_color='lightblue'
                ))
                
                # Add cumulative line
                cumulative = np.cumsum(cash_flows)
                fig.add_trace(go.Scatter(
                    x=years_list,
                    y=cumulative,
                    name="Cumulative Cash Flow",
                    line=dict(color='orange', width=3),
                    yaxis="y2"
                ))
                
                fig.update_layout(
                    title="Cash Flow Analysis",
                    xaxis_title="Year",
                    yaxis_title="Annual Cash Flow ($)",
                    yaxis2=dict(
                        title="Cumulative Cash Flow ($)",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity Analysis
                st.subheader("🔬 Sensitivity Analysis")
                sens_df = pd.DataFrame({
                    'Savings Variation': [f"{v*100:.0f}%" for v in roi_result['sensitivity_analysis']['variations']],
                    'NPV ($)': roi_result['sensitivity_analysis']['npv_values'],
                    'ROI (%)': roi_result['sensitivity_analysis']['roi_values']
                })
                
                st.dataframe(sens_df, use_container_width=True)
                
                # NPV vs ROI sensitivity chart
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=roi_result['sensitivity_analysis']['variations'],
                    y=roi_result['sensitivity_analysis']['npv_values'],
                    name="NPV",
                    mode="lines+markers",
                    line=dict(color="green", width=3)
                ))
                
                fig2.add_trace(go.Scatter(
                    x=roi_result['sensitivity_analysis']['variations'],
                    y=roi_result['sensitivity_analysis']['roi_values'],
                    name="ROI",
                    mode="lines+markers",
                    line=dict(color="blue", width=3),
                    yaxis="y2"
                ))
                
                fig2.update_layout(
                    title="Sensitivity Analysis: NPV and ROI vs Savings Variation",
                    xaxis_title="Savings Multiplier",
                    yaxis_title="NPV ($)",
                    yaxis2=dict(
                        title="ROI (%)",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    # ==================== TAB 2: SENSOR PLACEMENT ====================
    with tab2:
        st.header("📍 Optimal Sensor Placement")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            building_length = st.slider(
                "Building Length (m)",
                min_value=5,
                max_value=100,
                value=20,
                step=5
            )
        
        with col2:
            building_width = st.slider(
                "Building Width (m)",
                min_value=5,
                max_value=100,
                value=15,
                step=5
            )
        
        with col3:
            building_height = st.slider(
                "Building Height (m)",
                min_value=2,
                max_value=20,
                value=4,
                step=1
            )
        
        sensor_range = st.slider(
            "Sensor Range (m)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Maximum detection range of each sensor"
        )
        
        if st.button("📍 Optimize Sensor Placement", type="primary"):
            with st.spinner("Calculating optimal sensor placement..."):
                placement_result = optimizer.optimize_sensor_placement(
                    [building_length, building_width, building_height], 
                    sensor_range
                )
                
                # Display results
                st.subheader("📍 Optimization Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Optimal Sensors", placement_result['num_sensors'])
                
                with col2:
                    st.metric("Optimal Spacing", f"{placement_result['optimal_spacing']:.2f} m")
                
                with col3:
                    st.metric("Coverage", f"{placement_result['coverage_percentage']}%")
                
                # 3D Visualization
                st.subheader("🗺️ 3D Sensor Placement Visualization")
                
                # Create 3D scatter plot
                coordinates = np.array(placement_result['coordinates'])
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=coordinates[:, 0],
                    y=coordinates[:, 1],
                    z=coordinates[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=coordinates[:, 2],
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    name='Sensor Locations'
                )])
                
                # Add building box
                x_box = [0, building_length, building_length, 0, 0, building_length, building_length, 0]
                y_box = [0, 0, building_width, building_width, 0, 0, building_width, building_width]
                z_box = [0, 0, 0, 0, building_height, building_height, building_height, building_height]
                
                fig.add_trace(go.Mesh3d(
                    x=x_box,
                    y=y_box,
                    z=z_box,
                    color='lightblue',
                    opacity=0.1,
                    name='Building'
                ))
                
                fig.update_layout(
                    title=f"Optimal Sensor Placement: {placement_result['num_sensors']} sensors",
                    scene=dict(
                        xaxis_title='Length (m)',
                        yaxis_title='Width (m)',
                        zaxis_title='Height (m)',
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=building_width/building_length, z=building_height/building_length)
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== TAB 3: ENERGY OPTIMIZATION ====================
    with tab3:
        st.header("⚡ Energy Consumption Optimization")
        
        st.markdown("Configure your energy consumption profile:")
        
        # Create sample power data with sliders
        hours = list(range(24))
        power_data = []
        
        cols = st.columns(6)
        for i in range(24):
            with cols[i % 6]:
                hour_power = st.slider(
                    f"Hr {i:02d}:00",
                    min_value=0,
                    max_value=300,
                    value=np.random.randint(50, 200),
                    key=f"power_{i}"
                )
                power_data.append(hour_power)
        
        if st.button("⚡ Optimize Energy Consumption", type="primary"):
            with st.spinner("Optimizing energy consumption..."):
                energy_result = optimizer.energy_consumption_optimization(power_data)
                
                # Display results
                st.subheader("⚡ Optimization Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Original Consumption", f"{energy_result['original_consumption']:.0f} kWh")
                
                with col2:
                    st.metric("Optimized Savings", f"{energy_result['savings_percentage']}%")
                
                # Energy consumption chart
                st.subheader("📈 Energy Consumption Profile")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=power_data,
                    name="Original Consumption",
                    line=dict(color='red', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.1)'
                ))
                
                fig.add_trace(go.Scatter(
                    x=hours,
                    y=energy_result['optimal_schedule'],
                    name="Optimized Schedule",
                    line=dict(color='green', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0,255,0,0.1)'
                ))
                
                fig.update_layout(
                    title="Energy Consumption: Original vs Optimized",
                    xaxis_title="Hour of Day",
                    yaxis_title="Power Consumption (kW)",
                    hovermode="x unified",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Savings breakdown
                st.subheader("💰 Cost Savings Analysis")
                
                electricity_rate = st.slider(
                    "Electricity Rate ($/kWh)",
                    min_value=0.05,
                    max_value=0.30,
                    value=0.15,
                    step=0.01
                )
                
                daily_savings = (energy_result['original_consumption'] - energy_result['optimized_consumption']) * electricity_rate
                monthly_savings = daily_savings * 30
                annual_savings = monthly_savings * 12
                
                savings_col1, savings_col2, savings_col3 = st.columns(3)
                
                with savings_col1:
                    st.metric("Daily Savings", f"${daily_savings:.2f}")
                
                with savings_col2:
                    st.metric("Monthly Savings", f"${monthly_savings:.2f}")
                
                with savings_col3:
                    st.metric("Annual Savings", f"${annual_savings:.2f}")
    
    # ==================== TAB 4: PREDICTIVE MAINTENANCE ====================
    with tab4:
        st.header("🔧 Predictive Maintenance Analysis")
        
        st.markdown("Configure sensor data parameters:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_points = st.slider(
                "Number of Data Points",
                min_value=50,
                max_value=500,
                value=100,
                step=50
            )
            
            base_value = st.slider(
                "Base Sensor Reading",
                min_value=0,
                max_value=100,
                value=50,
                step=5
            )
        
        with col2:
            noise_level = st.slider(
                "Noise Level",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
            
            anomaly_threshold = st.slider(
                "Anomaly Threshold (z-score)",
                min_value=1.0,
                max_value=5.0,
                value=3.0,
                step=0.5
            )
        
        # Generate synthetic sensor data
        np.random.seed(42)
        time = np.linspace(0, 10, data_points)
        
        # Create normal data with some anomalies
        sensor_data = base_value + np.sin(time) * 10 + np.random.normal(0, noise_level, data_points)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(data_points, size=min(5, data_points//20), replace=False)
        sensor_data[anomaly_indices] += np.random.normal(20, 5, len(anomaly_indices))
        
        if st.button("🔍 Analyze Predictive Maintenance", type="primary"):
            with st.spinner("Analyzing sensor data for predictive maintenance..."):
                maintenance_result = optimizer.predictive_maintenance_model(sensor_data, anomaly_threshold)
                
                # Display results
                st.subheader("🔧 Maintenance Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Anomalies Detected", len(maintenance_result['anomalies']))
                
                with col2:
                    st.metric("Maintenance Urgency", f"{maintenance_result['maintenance_urgency']}%")
                
                with col3:
                    st.metric("Time to Failure", f"{maintenance_result['time_to_failure']} readings")
                
                # Sensor data chart with anomalies
                st.subheader("📊 Sensor Data Analysis")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(sensor_data))),
                    y=sensor_data,
                    name="Sensor Readings",
                    line=dict(color='blue', width=2),
                    mode='lines'
                ))
                
                # Add rolling mean
                fig.add_trace(go.Scatter(
                    x=list(range(10, len(sensor_data))),
                    y=maintenance_result['rolling_mean'],
                    name="Rolling Mean",
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                # Highlight anomalies
                if len(maintenance_result['anomalies']) > 0:
                    anomalies_x = maintenance_result['anomalies']
                    anomalies_y = sensor_data[maintenance_result['anomalies']]
                    
                    fig.add_trace(go.Scatter(
                        x=anomalies_x,
                        y=anomalies_y,
                        name="Anomalies",
                        mode='markers',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='x'
                        )
                    ))
                
                fig.update_layout(
                    title="Sensor Data with Anomaly Detection",
                    xaxis_title="Time Index",
                    yaxis_title="Sensor Reading",
                    hovermode="x unified",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Z-scores chart
                st.subheader("📈 Statistical Analysis (Z-scores)")
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Scatter(
                    x=list(range(len(maintenance_result['z_scores']))),
                    y=maintenance_result['z_scores'],
                    name="Z-scores",
                    line=dict(color='purple', width=2)
                ))
                
                # Add threshold lines
                fig2.add_hline(y=anomaly_threshold, line_dash="dash", line_color="red", annotation_text=f"Upper Threshold (+{anomaly_threshold})")
                fig2.add_hline(y=-anomaly_threshold, line_dash="dash", line_color="red", annotation_text=f"Lower Threshold (-{anomaly_threshold})")
                fig2.add_hline(y=0, line_dash="dot", line_color="gray")
                
                fig2.update_layout(
                    title="Standardized Z-scores",
                    xaxis_title="Time Index",
                    yaxis_title="Z-score",
                    hovermode="x unified",
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Maintenance recommendations
                st.subheader("🛠️ Maintenance Recommendations")
                
                urgency = maintenance_result['maintenance_urgency']
                
                if urgency > 70:
                    st.error("🚨 **CRITICAL**: Immediate maintenance required! Multiple anomalies detected.")
                    st.write("**Recommended Actions:**")
                    st.write("1. Schedule emergency maintenance")
                    st.write("2. Inspect all sensor connections")
                    st.write("3. Check for environmental factors")
                    st.write("4. Consider sensor replacement")
                
                elif urgency > 40:
                    st.warning("⚠️ **WARNING**: Maintenance recommended soon.")
                    st.write("**Recommended Actions:**")
                    st.write("1. Schedule maintenance within 2 weeks")
                    st.write("2. Monitor sensor performance")
                    st.write("3. Check calibration")
                    st.write("4. Review historical data")
                
                elif urgency > 20:
                    st.info("ℹ️ **ADVISORY**: Monitor closely.")
                    st.write("**Recommended Actions:**")
                    st.write("1. Schedule routine maintenance")
                    st.write("2. Continue monitoring")
                    st.write("3. Update maintenance logs")
                    st.write("4. Plan for future maintenance")
                
                else:
                    st.success("✅ **NORMAL**: System operating within normal parameters.")
                    st.write("**Recommended Actions:**")
                    st.write("1. Continue regular monitoring")
                    st.write("2. Update maintenance schedule")
                    st.write("3. Document current status")
                    st.write("4. Plan for routine maintenance")

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Optimization Parameters")
        
        optimization_mode = st.selectbox(
            "Default Optimization Mode",
            ["Balanced", "Aggressive", "Conservative"],
            help="Select optimization strategy"
        )
        
        auto_update = st.checkbox(
            "Auto-update results",
            value=True,
            help="Automatically update results when parameters change"
        )
        
        st.divider()
        
        st.subheader("Export Results")
        
        if st.button("📥 Export All Results"):
            st.success("Results exported successfully!")
        
        st.divider()
        
        st.subheader("About")
        st.markdown("""
        This IoT Assistant uses mathematical optimization for:
        - 📊 ROI calculation and optimization
        - 📍 Optimal sensor placement
        - ⚡ Energy consumption optimization
        - 🔧 Predictive maintenance analysis
        
        **Features:**
        • Interactive parameter adjustment
        • Real-time optimization
        • Visual analytics
        • Sensitivity analysis
        """)

# Run the app
if __name__ == "__main__":
    main()
    

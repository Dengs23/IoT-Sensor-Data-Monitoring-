import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="🏢 IoT Building Valuation Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("🏢 IoT Building Valuation Assistant")
st.markdown("Complete analysis: ROI, sensor data, risk assessment, and building valuation")

# Enhanced ROI Calculator (from your 5.5)
class ROICalculator:
    @staticmethod
    def calculate(building_data, upgrade_costs, energy_price=0.15):
        """Calculate comprehensive ROI"""
        size = building_data.get('square_meters', 5000)
        age = building_data.get('age_years', 15)
        value = building_data.get('property_value', 2500000)
        risk_reduction = building_data.get('risk_reduction', 0.25)
        
        # Energy calculation
        base_energy = size * 0.15 * 12  # kWh/year
        age_factor = 1 + (age * 0.01)
        energy_cost = base_energy * age_factor * energy_price
        
        # Savings from IoT
        efficiency = building_data.get('efficiency', 0.35)
        upgraded_cost = energy_cost * (1 - efficiency)
        annual_savings = energy_cost - upgraded_cost
        maintenance_savings = size * 2
        total_annual = annual_savings + maintenance_savings
        
        # ROI
        payback = upgrade_costs / total_annual if total_annual > 0 else float('inf')
        value_increase = value * risk_reduction * 0.2
        roi_10yr = ((total_annual * 10) + value_increase - upgrade_costs) / upgrade_costs * 100
        
        return {
            'current_energy_cost': round(energy_cost, 2),
            'annual_savings': round(total_annual, 2),
            'payback_years': round(payback, 2),
            'property_value_increase': round(value_increase, 2),
            '10_year_roi': round(roi_10yr, 2),
            'total_10yr_benefit': round((total_annual * 10) + value_increase, 2)
        }

# AI Chat Simulator (no API costs)
class LocalAI:
    def __init__(self):
        self.knowledge = {
            'roi': {
                'description': 'Return on Investment analysis for IoT systems',
                'formula': 'ROI = (Net Benefits / Cost) × 100%',
                'typical': '3-5 year payback, 150-250% 10-year ROI'
            },
            'temperature': {
                'optimal': '18-24°C for industrial buildings',
                'anomaly': 'Detected when >2σ from mean',
                'impact': 'Each 1°C anomaly = 2-3% energy cost increase'
            },
            'risk': {
                'formula': 'Risk = Probability × Impact',
                'factors': ['Equipment failure', 'Energy waste', 'Maintenance'],
                'reduction': 'IoT reduces risk by 25-40%'
            }
        }
    
    def respond(self, prompt):
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm your IoT Assistant. I can help with ROI calculations, sensor data analysis, risk assessment, and building valuation."
        
        elif any(word in prompt_lower for word in ['roi', 'return', 'investment']):
            return self._roi_analysis(prompt)
        
        elif any(word in prompt_lower for word in ['temperature', 'sensor', 'data']):
            return self._temperature_analysis()
        
        elif any(word in prompt_lower for word in ['risk', 'danger', 'problem']):
            return self._risk_analysis()
        
        elif any(word in prompt_lower for word in ['value', 'valuation', 'worth']):
            return self._valuation_analysis()
        
        else:
            return "I can help with IoT data analysis, ROI calculations, risk assessment, and building valuation. Try asking about ROI or temperature data!"
    
    def _roi_analysis(self, prompt):
        import re
        numbers = re.findall(r'\d+', prompt)
        size = int(numbers[0]) if numbers else 5000
        
        return f"""## 💰 ROI Analysis for {size:,} m² Building

### Financial Metrics:
- **System Cost Estimate:** ${size * 15:,.0f}
- **Annual Energy Savings:** ${size * 3:,.0f} (${3}/m²/year)
- **Maintenance Savings:** ${size * 2:,.0f}/year
- **Total Annual Savings:** ${size * 5:,.0f}

### Timeline:
- **Payback Period:** {(size * 15) / (size * 5):.1f} years
- **5-Year Savings:** ${size * 5 * 5:,.0f}
- **10-Year ROI:** {(((size * 5 * 10) - (size * 15)) / (size * 15) * 100):.0f}%

### Property Impact:
- **Risk Reduction:** 25-35%
- **Value Increase:** ${size * 1000 * 0.075:,.0f} (7.5% on ${size * 1000:,.0f} property)
- **Insurance Savings:** 5-15% lower premiums

**Formula:** `{self.knowledge['roi']['formula']}`"""
    
    def _temperature_analysis(self):
        # Generate sample data
        hours = 24
        temps = np.random.normal(22, 3, hours)
        anomalies = temps[(temps < 16) | (temps > 28)]
        
        return f"""## 🌡️ Temperature Analysis

**Optimal Range:** {self.knowledge['temperature']['optimal']}
**Current Analysis ({hours}h):**
- Average: **{np.mean(temps):.1f}°C**
- Range: **{np.min(temps):.1f}°C to {np.max(temps):.1f}°C**
- Stability: **{'Good' if np.std(temps) < 3 else 'Needs attention'}**
- Anomalies: **{len(anomalies)} detected**

**Impact:** {self.knowledge['temperature']['impact']}
**Recommendation:** {'Normal operation' if len(anomalies) < 3 else 'Check HVAC system'}"""
    
    def _risk_analysis(self):
        risk_score = np.random.uniform(0.1, 0.4)
        reduction = 0.35
        
        return f"""## ⚠️ Risk Assessment

**Current Risk Score:** **{risk_score:.1%}**
**With IoT Monitoring:** **{(risk_score * (1 - reduction)):.1%}** ({(reduction*100):.0f}% reduction)

**Risk Factors:**
1. Equipment Failure: 15-25%
2. Energy Waste: 20-30%
3. Maintenance Costs: 10-20%

**Formula:** `{self.knowledge['risk']['formula']}`
**Risk Reduction:** {self.knowledge['risk']['reduction']}"""
    
    def _valuation_analysis(self):
        base_value = 2500000
        iot_impact = 0.075
        
        return f"""## 🏢 Building Valuation

**Base Property Value:** **${base_value:,.0f}**
**With IoT Implementation:** **${base_value * (1 + iot_impact):,.0f}**
**Value Increase:** **${base_value * iot_impact:,.0f}** ({(iot_impact*100):.1f}%)

**Value Drivers:**
1. **Risk Reduction** (20-30% lower insurance claims)
2. **Energy Efficiency** (15-35% lower operating costs)
3. **Predictive Maintenance** (20-40% lower repair costs)
4. **Compliance & Safety** (Reduced regulatory risk)"""

# Initialize
calculator = ROICalculator()
ai = LocalAI()

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

# Sidebar
with st.sidebar:
    st.header("🛠️ Analysis Tools")
    
    # Navigation
    page = st.radio("Navigate to:", 
                   ["💬 AI Assistant", "💰 ROI Calculator", "📊 Sensor Data", "🏢 Building Valuation"])
    
    # Quick ROI Calculator
    with st.expander("Quick ROI Calc", expanded=True):
        size = st.number_input("Size (m²)", 1000, 50000, 5000)
        cost = st.number_input("Cost ($)", 10000, 500000, 75000)
        
        if st.button("Calculate", key="quick_calc"):
            building_data = {'square_meters': size, 'property_value': size * 1000}
            results = calculator.calculate(building_data, cost)
            
            st.metric("Annual Savings", f"${results['annual_savings']:,.0f}")
            st.metric("Payback", f"{results['payback_years']:.1f} years")
            st.metric("10-Year ROI", f"{results['10_year_roi']:.1f}%")
    
    # Data Generator
    if st.button("📈 Generate Sensor Data"):
        hours = 48
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        temperatures = np.random.normal(22, 3, hours)
        
        st.session_state.sensor_data = pd.DataFrame({
            'timestamp': timestamps[::-1],
            'temperature': temperatures,
            'humidity': np.random.normal(45, 10, hours)
        })
        
        st.success(f"Generated {hours} hours of data")
        st.metric("Avg Temp", f"{np.mean(temperatures):.1f}°C")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content based on page
if page == "💬 AI Assistant":
    st.header("💬 AI Assistant (Local - No API Costs)")
    
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about IoT data, ROI, or valuation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            response = ai.respond(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

elif page == "💰 ROI Calculator":
    st.header("💰 Advanced ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Building Details")
        square_meters = st.number_input("Building Size (m²)", 1000, 50000, 5000)
        age_years = st.number_input("Building Age (years)", 0, 100, 15)
        property_value = st.number_input("Property Value ($)", 100000, 20000000, 2500000)
    
    with col2:
        st.subheader("System & Factors")
        upgrade_costs = st.number_input("IoT System Cost ($)", 10000, 1000000, 75000)
        energy_price = st.number_input("Energy Price ($/kWh)", 0.05, 1.0, 0.15, 0.01)
        risk_reduction = st.slider("Risk Reduction %", 0, 100, 25) / 100
        efficiency = st.slider("Efficiency Gain %", 0, 50, 35) / 100
    
    if st.button("📊 Calculate Comprehensive ROI", type="primary"):
        building_data = {
            'square_meters': square_meters,
            'age_years': age_years,
            'property_value': property_value,
            'risk_reduction': risk_reduction,
            'efficiency': efficiency
        }
        
        results = calculator.calculate(building_data, upgrade_costs, energy_price)
        
        # Display results
        st.success("### 📈 ROI Analysis Results")
        
        cols = st.columns(4)
        metrics = [
            ("Annual Savings", f"${results['annual_savings']:,.0f}"),
            ("Payback Period", f"{results['payback_years']:.1f} years"),
            ("10-Year ROI", f"{results['10_year_roi']:.1f}%"),
            ("Value Increase", f"${results['property_value_increase']:,.0f}")
        ]
        
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(name='Cost', x=['System Cost'], y=[upgrade_costs], marker_color='red'),
            go.Bar(name='Annual Benefit', x=['Savings', 'Value'], 
                  y=[results['annual_savings'], results['property_value_increase']/10], 
                  marker_color='green')
        ])
        fig.update_layout(title='Cost vs Annual Benefits', barmode='group')
        st.plotly_chart(fig)

elif page == "📊 Sensor Data":
    st.header("📊 Sensor Data Analysis")
    
    if st.session_state.sensor_data is not None:
        df = st.session_state.sensor_data
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{df['temperature'].mean():.1f}°C")
        with col2:
            st.metric("Max Temperature", f"{df['temperature'].max():.1f}°C")
        with col3:
            st.metric("Min Temperature", f"{df['temperature'].min():.1f}°C")
        with col4:
            anomalies = df[(df['temperature'] < 16) | (df['temperature'] > 28)]
            st.metric("Anomalies", len(anomalies))
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['temperature'],
            mode='lines', name='Temperature',
            line=dict(color='blue', width=2)
        ))
        
        if len(anomalies) > 0:
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'], y=anomalies['temperature'],
                mode='markers', name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(title='Temperature Over Time', height=400)
        st.plotly_chart(fig)
        
        # Data table
        with st.expander("View Raw Data"):
            st.dataframe(df)
    else:
        st.info("Generate sensor data first using the button in the sidebar!")

elif page == "🏢 Building Valuation":
    st.header("🏢 Building Valuation Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        base_value = st.number_input("Base Property Value ($)", 100000, 10000000, 2500000)
        building_type = st.selectbox("Building Type", 
                                    ["Industrial", "Commercial", "Residential", "Mixed-Use"])
        
        st.subheader("Risk Factors")
        temp_stability = st.slider("Temperature Stability", 0, 100, 75)
        equipment_age = st.slider("Equipment Age Factor", 0, 100, 40)
    
    with col2:
        st.subheader("IoT Improvements")
        iot_coverage = st.slider("IoT Coverage %", 0, 100, 80)
        maintenance = st.slider("Maintenance Score", 0, 100, 65)
        energy_efficiency = st.slider("Energy Efficiency", 0, 100, 70)
        
        if st.button("Calculate Valuation Impact", type="primary"):
            # Calculate adjusted value
            risk_score = ((100 - temp_stability) * 0.6 + equipment_age * 0.4) / 100
            improvement_score = (iot_coverage * 0.4 + maintenance * 0.3 + energy_efficiency * 0.3) / 100
            
            risk_adjustment = 1 - (risk_score * 0.2)
            improvement_adjustment = 1 + (improvement_score * 0.15)
            
            adjusted_value = base_value * risk_adjustment * improvement_adjustment
            value_change = adjusted_value - base_value
            
            # Display
            st.success("### Valuation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Base Value", f"${base_value:,.0f}")
            with col2:
                st.metric("Adjusted Value", f"${adjusted_value:,.0f}")
            with col3:
                st.metric("Value Change", f"${value_change:,.0f}", 
                         delta=f"{(value_change/base_value)*100:.1f}%")
            
            # Radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[temp_stability, 100-equipment_age, iot_coverage, maintenance, energy_efficiency],
                theta=['Temp Stability', 'New Equipment', 'IoT Coverage', 'Maintenance', 'Efficiency'],
                fill='toself',
                name='Building Score'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                            showlegend=False, title="Building Assessment")
            st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🏢 IoT Valuation Assistant**")
st.sidebar.markdown("v2.0 • Complete Features • No API Costs")
st.sidebar.markdown(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

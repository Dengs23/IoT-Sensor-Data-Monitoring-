import streamlit as st
import numpy as np
from datetime import datetime

st.set_page_config(page_title="IoT Assistant", layout="wide")
st.title("🏢 IoT Building Valuation Assistant")
st.markdown("Smart analysis for ROI calculations and sensor data")

# Simple ROI calculator
def calculate_simple_roi(size_m2, cost):
    annual = size_m2 * 3
    payback = cost / annual
    roi = ((annual * 10) - cost) / cost * 100
    return annual, payback, roi

# Initialize chat
if "chat" not in st.session_state:
    st.session_state.chat = []

# Sidebar tools
with st.sidebar:
    st.header("🛠️ Tools")
    
    # ROI Calculator
    size = st.number_input("Building Size (m²)", 1000, 50000, 5000)
    cost = st.number_input("System Cost ($)", 10000, 500000, 75000)
    
    if st.button("💰 Calculate ROI"):
        annual, payback, roi = calculate_simple_roi(size, cost)
        st.metric("Annual Savings", f"${annual:,.0f}")
        st.metric("Payback", f"{payback:.1f} years")
        st.metric("10-Year ROI", f"{roi:.1f}%")
        
        st.session_state.chat.append(f"ROI for {size:,}m²: ${annual:,.0f}/year savings, {payback:.1f} year payback")
    
    # Data generator
    if st.button("📊 Generate Data"):
        temps = np.random.normal(22, 3, 24)
        st.metric("Avg Temp", f"{np.mean(temps):.1f}°C")
        st.session_state.chat.append(f"Generated sensor data: {np.mean(temps):.1f}°C avg")
    
    # Clear
    if st.button("🗑️ Clear"):
        st.session_state.chat = []

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Analysis")
    
    for msg in st.session_state.chat:
        st.info(msg)
    
    # Quick questions
    question = st.selectbox(
        "Ask a question:",
        ["Select...", "What's typical ROI?", "How does IoT affect value?", "Temperature guidelines?"]
    )
    
    if question != "Select...":
        if "ROI" in question:
            st.success("💰 Typical IoT ROI: 150-250% over 10 years, 3-5 year payback")
        elif "value" in question.lower():
            st.success("🏢 Value increase: 5-15% from risk reduction and efficiency")
        elif "temperature" in question.lower():
            st.success("🌡️ Optimal: 18-24°C. Each 1°C anomaly increases energy cost 2-3%")

with col2:
    st.header("📊 Live Data")
    st.metric("Current Temp", f"{22.5 + np.random.randn():.1f}°C")
    st.metric("Humidity", f"{45 + np.random.randn()*5:.1f}%")
    st.metric("Time", datetime.now().strftime("%H:%M"))
    st.metric("Status", "✅ Active")

st.sidebar.markdown("---")
st.sidebar.markdown("**IoT Assistant** • Streamlit Cloud")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(
    page_title="🤖 IoT AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IoT Data AI Assistant")
st.markdown("Smart analysis of sensor data with ROI calculations")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I can help analyze IoT data and calculate ROI. Ask me anything!"}
    ]

# Sidebar with tools
with st.sidebar:
    st.header("🛠️ Tools")
    
    # Sensor Data Generator
    if st.button("📊 Generate Sensor Data"):
        hours = st.slider("Hours", 1, 72, 24)
        
        # Generate data
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        temps = np.random.normal(22, 3, hours)
        
        # Store in session
        st.session_state.sensor_data = {
            'timestamps': timestamps[::-1],
            'temperatures': temps,
            'hours': hours
        }
        
        st.success(f"Generated {hours} hours of data")
        st.metric("Avg Temp", f"{np.mean(temps):.1f}°C")
    
    # ROI Calculator
    with st.expander("💰 ROI Calculator", expanded=True):
        size = st.number_input("Building Size (m²)", 1000, 20000, 5000)
        cost = st.number_input("System Cost ($)", 10000, 200000, 75000)
        
        if st.button("Calculate ROI"):
            annual_savings = size * 3
            payback = cost / annual_savings
            roi = ((annual_savings * 10) - cost) / cost * 100
            
            st.metric("Annual Savings", f"${annual_savings:,.0f}")
            st.metric("Payback", f"{payback:.1f} years")
            st.metric("10-Year ROI", f"{roi:.1f}%")
            
            # Add to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"ROI Analysis:\n- Annual Savings: ${annual_savings:,.0f}\n- Payback: {payback:.1f} years\n- 10-Year ROI: {roi:.1f}%"
            })
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat area
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Conversation")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about sensor data or ROI..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = ""
        if "temperature" in prompt.lower() or "sensor" in prompt.lower():
            if 'sensor_data' in st.session_state:
                data = st.session_state.sensor_data
                avg_temp = np.mean(data['temperatures'])
                response = f"📊 Sensor Data Analysis:\nAverage temperature: {avg_temp:.1f}°C\nHours of data: {data['hours']}\n\nTry asking about ROI or risk assessment!"
            else:
                response = "I can analyze sensor data! First, generate some data using the 'Generate Sensor Data' button in the sidebar."
        
        elif "roi" in prompt.lower() or "return" in prompt.lower():
            response = "💰 ROI Information:\nTypical IoT implementations show:\n- Payback: 3-5 years\n- Annual Savings: $15-25 per m²\n- 10-Year ROI: 150-250%\n\nUse the ROI Calculator in the sidebar for specific numbers!"
        
        elif "risk" in prompt.lower():
            response = "⚠️ Risk Assessment:\nBased on typical sensor data:\n- Equipment Failure Risk: 15-25%\n- Energy Waste Risk: 20-30%\n- Maintenance Risk: 10-20%\n\nIoT monitoring reduces these risks by 25-40%."
        
        else:
            response = "I can help you with:\n1. 📊 Sensor data analysis\n2. 💰 ROI calculations\n3. ⚠️ Risk assessment\n4. 🏢 Building valuation\n\nTry asking about temperature data or ROI!"
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    st.header("📈 Quick Stats")
    
    # Show some statistics
    if 'sensor_data' in st.session_state:
        data = st.session_state.sensor_data
        temps = data['temperatures']
        
        st.metric("Avg Temp", f"{np.mean(temps):.1f}°C")
        st.metric("Max Temp", f"{np.max(temps):.1f}°C")
        st.metric("Min Temp", f"{np.min(temps):.1f}°C")
        st.metric("Data Points", len(temps))
    else:
        st.info("Generate sensor data first!")
    
    # Quick actions
    st.header("⚡ Quick Actions")
    if st.button("🔄 Refresh"):
        st.rerun()
    
    if st.button("📥 Export"):
        st.success("Export ready!")

st.sidebar.markdown("---")
st.sidebar.markdown("**IoT AI Assistant v1.0**")
st.sidebar.markdown("Smart analysis for building management")

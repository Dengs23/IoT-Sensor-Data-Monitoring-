import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import streamlit as st

# Try different import patterns based on LangChain version
try:
    # Newer LangChain versions
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_openai import ChatOpenAI
    LANGCHAIN_NEW = True
except ImportError:
    try:
        # Older LangChain versions
        from langchain.agents import initialize_agent, AgentType
        from langchain.chat_models import ChatOpenAI
        LANGCHAIN_NEW = False
    except ImportError:
        # LangChain not installed or different structure
        st.error("Please install LangChain: pip install langchain langchain-openai")
        st.stop()

from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ROI Calculator Functions (same as before)
def estimate_energy_usage(building_data: Dict[str, Any]) -> float:
    """Estimate energy usage based on building characteristics"""
    base_usage = building_data.get('square_meters', 1000) * 0.15
    
    age_factor = 1 + (building_data.get('age_years', 10) * 0.01)
    insulation = building_data.get('insulation_rating', 5)
    insulation_factor = 1.2 - (insulation * 0.02)
    temp_variance = building_data.get('temperature_variance', 0)
    temp_factor = 1 + (temp_variance * 0.1)
    
    return base_usage * age_factor * insulation_factor * temp_factor

def calculate_roi_simple(**kwargs) -> Dict[str, Any]:
    """Simplified ROI calculator"""
    square_meters = kwargs.get('square_meters', 5000)
    upgrade_costs = kwargs.get('upgrade_costs', 75000)
    property_value = kwargs.get('property_value', 2500000)
    
    # Simple calculation
    annual_savings = square_meters * 3  # $3 per m² per year
    payback_years = upgrade_costs / annual_savings
    roi_10yr = ((annual_savings * 10) - upgrade_costs) / upgrade_costs * 100
    value_increase = property_value * 0.05  # 5% increase
    
    return {
        'annual_savings': round(annual_savings, 2),
        'payback_years': round(payback_years, 1),
        'roi_10yr': round(roi_10yr, 1),
        'value_increase': round(value_increase, 2),
        'total_benefit_10yr': round((annual_savings * 10) + value_increase, 2)
    }

# IoT Tools
class IoTAssistant:
    def __init__(self):
        self.sensor_data_cache = None
        
    def get_sensor_data(self, hours: int = 24) -> Dict[str, Any]:
        """Get simulated sensor data"""
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        temperatures = np.random.normal(22, 3, hours)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(hours, size=max(1, hours//20), replace=False)
        for idx in anomaly_indices:
            temperatures[idx] += np.random.choice([-10, 12])
        
        self.sensor_data_cache = {
            'timestamps': [ts.isoformat() for ts in timestamps[::-1]],
            'temperatures': temperatures.tolist(),
            'hours': hours
        }
        
        return {
            'average_temp': float(np.mean(temperatures)),
            'max_temp': float(np.max(temperatures)),
            'min_temp': float(np.min(temperatures)),
            'anomalies': len([t for t in temperatures if abs(t - 22) > 8]),
            'message': f"Generated {hours} hours of sensor data"
        }
    
    def analyze_risk(self, temperature_data: List[float] = None) -> Dict[str, Any]:
        """Analyze risk based on temperature data"""
        if temperature_data is None:
            if self.sensor_data_cache:
                temps = self.sensor_data_cache['temperatures']
            else:
                temps = np.random.normal(22, 3, 100).tolist()
        else:
            temps = temperature_data
        
        temps_array = np.array(temps)
        mean_temp = np.mean(temps_array)
        std_temp = np.std(temps_array)
        
        # Count anomalies
        anomalies = temps_array[(temps_array < mean_temp - 2*std_temp) | (temps_array > mean_temp + 2*std_temp)]
        
        risk_score = min(1.0, len(anomalies) / len(temps) * 3)
        
        return {
            'risk_score': round(float(risk_score), 3),
            'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
            'anomaly_count': int(len(anomalies)),
            'temperature_stability': 'Unstable' if std_temp > 4 else 'Moderate' if std_temp > 2 else 'Stable',
            'recommendation': 'Immediate action needed' if risk_score > 0.7 else 'Monitor closely' if risk_score > 0.3 else 'Normal operation'
        }

# Streamlit App
def main():
    st.set_page_config(
        page_title="🤖 IoT AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 IoT Data & ROI AI Assistant")
    st.markdown("Ask questions about sensor data, ROI calculations, and building valuation")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = IoTAssistant()
    
    # Initialize chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar with tools
    with st.sidebar:
        st.header("🛠️ Tools")
        
        # Sensor Data Tool
        if st.button("📊 Generate Sensor Data"):
            hours = st.slider("Hours of data", 1, 168, 24, key="data_hours")
            data = st.session_state.assistant.get_sensor_data(hours)
            
            st.success(f"✅ Generated {hours} hours of data")
            st.metric("Average Temp", f"{data['average_temp']:.1f}°C")
            st.metric("Anomalies", data['anomalies'])
            
            # Store for chat
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"I've generated {hours} hours of sensor data. Average temperature: {data['average_temp']:.1f}°C with {data['anomalies']} anomalies detected."
            })
        
        # ROI Calculator Tool
        with st.expander("💰 ROI Calculator", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                size = st.number_input("Building Size (m²)", 1000, 50000, 5000, key="roi_size")
                cost = st.number_input("Upgrade Cost ($)", 10000, 500000, 75000, step=5000, key="roi_cost")
            with col2:
                value = st.number_input("Property Value ($)", 100000, 10000000, 2500000, step=100000, key="roi_value")
            
            if st.button("Calculate", key="calc_roi"):
                result = calculate_roi_simple(
                    square_meters=size,
                    upgrade_costs=cost,
                    property_value=value
                )
                
                st.metric("Annual Savings", f"${result['annual_savings']:,.0f}")
                st.metric("Payback", f"{result['payback_years']} years")
                st.metric("10-Year ROI", f"{result['roi_10yr']}%")
                
                # Store for chat
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': f"""ROI Analysis:
- Annual Savings: ${result['annual_savings']:,.0f}
- Payback Period: {result['payback_years']} years
- 10-Year ROI: {result['roi_10yr']}%
- Value Increase: ${result['value_increase']:,.0f}"""
                })
        
        # Risk Analysis Tool
        if st.button("⚠️ Analyze Risk"):
            risk_data = st.session_state.assistant.analyze_risk()
            
            st.warning(f"Risk Level: {risk_data['risk_level']}")
            st.metric("Risk Score", f"{risk_data['risk_score']:.1%}")
            st.metric("Anomalies", risk_data['anomaly_count'])
            
            # Store for chat
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"""Risk Analysis:
- Risk Level: {risk_data['risk_level']}
- Risk Score: {risk_data['risk_score']:.1%}
- Temperature Stability: {risk_data['temperature_stability']}
- Recommendation: {risk_data['recommendation']}"""
            })
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Conversation")
        
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
        
        # Chat input
        if prompt := st.chat_input("Ask about IoT data, ROI, or risks..."):
            # Add user message
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    # Simple rule-based responses (in a real app, this would be LangChain)
                    response = generate_response(prompt, st.session_state.assistant)
                    st.write(response)
            
            # Add assistant response
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Always show some stats
        data = st.session_state.assistant.get_sensor_data(24)
        risk = st.session_state.assistant.analyze_risk()
        
        st.metric("Avg Temperature", f"{data['average_temp']:.1f}°C")
        st.metric("Risk Level", risk['risk_level'])
        st.metric("Data Points", "1,000+")
        
        st.progress(risk['risk_score'], text=f"Risk Score: {risk['risk_score']:.1%}")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        if st.button("📥 Export Report"):
            report = {
                'timestamp': datetime.now().isoformat(),
                'sensor_data': data,
                'risk_analysis': risk
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(report, indent=2),
                file_name=f"iot_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

def generate_response(prompt: str, assistant: IoTAssistant) -> str:
    """Generate response based on user prompt"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm your IoT Assistant. I can help you with sensor data analysis, ROI calculations, and risk assessment."
    
    elif any(word in prompt_lower for word in ['sensor', 'data', 'temperature', 'reading']):
        data = assistant.get_sensor_data(24)
        return f"""**📊 Sensor Data Summary:**
- Average Temperature: {data['average_temp']:.1f}°C
- Temperature Range: {data['min_temp']:.1f}°C to {data['max_temp']:.1f}°C
- Anomalies Detected: {data['anomalies']}
- Data Coverage: 24 hours

Use the tools in the sidebar to generate more data or analyze risks."""
    
    elif any(word in prompt_lower for word in ['roi', 'return', 'investment', 'payback']):
        return """**💰 ROI Information:**
I can calculate ROI for IoT implementations. Typical results show:
- Payback Period: 3-5 years
- Annual Energy Savings: $15-25 per m²
- Property Value Increase: 5-15%
- 10-Year ROI: 150-300%

Use the ROI Calculator in the sidebar for specific calculations."""
    
    elif any(word in prompt_lower for word in ['risk', 'anomaly', 'problem', 'issue']):
        risk = assistant.analyze_risk()
        return f"""**⚠️ Risk Analysis:**
- Current Risk Level: **{risk['risk_level']}**
- Risk Score: {risk['risk_score']:.1%}
- Temperature Stability: {risk['temperature_stability']}
- Anomalies in data: {risk['anomaly_count']}
- **Recommendation:** {risk['recommendation']}

High risk usually indicates frequent temperature fluctuations that could affect building systems."""
    
    elif any(word in prompt_lower for word in ['value', 'valuation', 'worth', 'price']):
        return """**🏢 Building Valuation Impact:**
IoT monitoring typically increases property value by 5-15% through:
1. **Risk Reduction** (20-30% lower failure risk)
2. **Energy Efficiency** (15-35% energy savings)
3. **Predictive Maintenance** (20-40% lower costs)
4. **Insurance Benefits** (5-15% lower premiums)

A $2.5M building could see a $125,000-$375,000 value increase."""
    
    elif any(word in prompt_lower for word in ['help', 'what can you do', 'capabilities']):
        return """**I can help you with:**
1. **📊 Sensor Data Analysis** - Temperature, anomalies, trends
2. **💰 ROI Calculations** - Investment returns for IoT systems
3. **⚠️ Risk Assessment** - Identify potential issues
4. **🏢 Building Valuation** - Estimate property value impact
5. **📈 Reporting** - Generate analysis reports

Try asking:
- "Show me sensor data"
- "Calculate ROI for my building"
- "What's the risk level?"
- "How does IoT affect property value?" """
    
    else:
        return "I can help you with IoT data analysis, ROI calculations, and risk assessment. Could you be more specific about what you'd like to know?"

if __name__ == "__main__":
    main()

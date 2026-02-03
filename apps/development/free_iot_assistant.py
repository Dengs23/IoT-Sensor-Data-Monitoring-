import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="🤖 Free IoT AI Assistant",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 IoT Smart Assistant (Free Local Version)")
st.markdown("No API keys needed - Intelligent analysis with local processing")

# Enhanced rule-based AI
class LocalAI:
    def __init__(self):
        self.knowledge_base = {
            'roi': {
                'calculation': "ROI = (Net Gain / Cost) × 100%",
                'typical': "3-5 year payback, 150-250% 10-year ROI",
                'factors': ["Energy savings", "Maintenance reduction", "Risk mitigation", "Value increase"]
            },
            'temperature': {
                'optimal': "18-24°C for industrial buildings",
                'anomaly': ">2 standard deviations from mean",
                'impact': "Each 1°C anomaly increases energy cost by 2-3%"
            },
            'risk': {
                'formula': "Risk = Probability × Impact",
                'reduction': "IoT reduces risk by 25-40%",
                'components': ["Equipment failure", "Energy waste", "Maintenance costs"]
            }
        }
    
    def analyze(self, prompt):
        prompt_lower = prompt.lower()
        
        # Check for building parameters
        import re
        numbers = re.findall(r'\d+', prompt)
        
        response = "## 🤖 Analysis Results\n\n"
        
        # ROI analysis
        if any(word in prompt_lower for word in ['roi', 'return', 'investment', 'payback']):
            response += "### 💰 ROI Analysis\n\n"
            
            if numbers:
                size = int(numbers[0]) if len(numbers) > 0 else 5000
                cost = int(numbers[1]) if len(numbers) > 1 else 75000
                
                annual_savings = size * 3
                payback = cost / annual_savings
                roi_10yr = ((annual_savings * 10) - cost) / cost * 100
                
                response += f"**For {size:,} m² building with ${cost:,} investment:**\n"
                response += f"- Annual Savings: **${annual_savings:,.0f}**\n"
                response += f"- Payback Period: **{payback:.1f} years**\n"
                response += f"- 10-Year ROI: **{roi_10yr:.1f}%**\n"
                response += f"- Value Increase: **${(size * 50):,.0f}** (est.)\n\n"
            
            response += f"**Formula:** `{self.knowledge_base['roi']['calculation']}`\n"
            response += f"**Typical Results:** {self.knowledge_base['roi']['typical']}\n"
            response += "**Key Factors:** " + ", ".join(self.knowledge_base['roi']['factors'])
        
        # Temperature analysis
        elif any(word in prompt_lower for word in ['temp', 'heat', 'cold', 'sensor']):
            response += "### 🌡️ Temperature Analysis\n\n"
            
            # Generate sample data
            hours = 24
            temps = np.random.normal(22, 3, hours)
            
            response += f"**Optimal Range:** {self.knowledge_base['temperature']['optimal']}\n"
            response += f"**Current Analysis ({hours}h):**\n"
            response += f"- Average: **{np.mean(temps):.1f}°C**\n"
            response += f"- Range: **{np.min(temps):.1f}°C to {np.max(temps):.1f}°C**\n"
            
            anomalies = temps[(temps < 16) | (temps > 28)]
            response += f"- Anomalies: **{len(anomalies)} detected**\n"
            response += f"- Stability: **{'Good' if np.std(temps) < 3 else 'Needs attention'}**\n\n"
            
            response += f"**Impact:** {self.knowledge_base['temperature']['impact']}\n"
            response += "**Recommendation:** " + ("Normal operation" if len(anomalies) < 3 else "Check HVAC system")
        
        # Risk analysis
        elif any(word in prompt_lower for word in ['risk', 'danger', 'problem', 'issue']):
            response += "### ⚠️ Risk Assessment\n\n"
            
            risk_score = np.random.uniform(0.1, 0.4)
            reduction = 0.35
            
            response += f"**Current Risk Score:** **{risk_score:.1%}**\n"
            response += f"**With IoT Monitoring:** **{(risk_score * (1 - reduction)):.1%}** ({(reduction*100):.0f}% reduction)\n\n"
            
            response += f"**Formula:** `{self.knowledge_base['risk']['formula']}`\n"
            response += f"**Risk Reduction:** {self.knowledge_base['risk']['reduction']}\n"
            response += "**Components:** " + ", ".join(self.knowledge_base['risk']['components'])
        
        # Building valuation
        elif any(word in prompt_lower for word in ['value', 'valuation', 'worth', 'price']):
            response += "### 🏢 Building Valuation\n\n"
            
            base_value = 2500000
            iot_impact = 0.075  # 7.5% increase
            
            response += f"**Base Property Value:** **${base_value:,.0f}**\n"
            response += f"**With IoT Implementation:** **${base_value * (1 + iot_impact):,.0f}**\n"
            response += f"**Value Increase:** **${base_value * iot_impact:,.0f}** ({(iot_impact*100):.1f}%)\n\n"
            
            response += "**Value Drivers:**\n"
            response += "1. **Risk Reduction** (20-30% lower insurance claims)\n"
            response += "2. **Energy Efficiency** (15-35% lower operating costs)\n"
            response += "3. **Predictive Maintenance** (20-40% lower repair costs)\n"
            response += "4. **Compliance & Safety** (Reduced regulatory risk)\n"
        
        else:
            response = """## 🎯 IoT Assistant Help

I can analyze:
1. **💰 ROI Calculations** - Return on investment for IoT systems
2. **🌡️ Temperature Data** - Sensor readings and anomalies  
3. **⚠️ Risk Assessment** - Building risk scores and reduction
4. **🏢 Building Valuation** - Property value impact

**Example questions:**
- "Calculate ROI for 5000 m² building with $75k investment"
- "Analyze temperature data for risk assessment"
- "How much does IoT increase building value?"
- "What's the payback period for sensors?"

Try asking a specific question!"""
        
        return response

# Initialize AI
ai = LocalAI()

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your free IoT AI Assistant. I can analyze ROI, temperature data, risks, and building valuation. Ask me anything!"}
    ]

# Sidebar tools
with st.sidebar:
    st.header("🛠️ Quick Tools")
    
    # ROI Calculator
    with st.expander("💰 ROI Calculator", expanded=True):
        size = st.number_input("Building Size (m²)", 1000, 50000, 5000)
        cost = st.number_input("System Cost ($)", 10000, 500000, 75000)
        
        if st.button("Calculate", key="roi_calc"):
            annual_savings = size * 3
            payback = cost / annual_savings
            roi = ((annual_savings * 10) - cost) / cost * 100
            
            st.metric("Annual Savings", f"${annual_savings:,.0f}")
            st.metric("Payback", f"{payback:.1f} years")
            st.metric("10-Year ROI", f"{roi:.1f}%")
    
    # Data Generator
    if st.button("📊 Generate Sample Data"):
        hours = 24
        temps = np.random.normal(22, 3, hours)
        
        st.metric("Avg Temp", f"{np.mean(temps):.1f}°C")
        st.metric("Max Temp", f"{np.max(temps):.1f}°C")
        st.metric("Min Temp", f"{np.min(temps):.1f}°C")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Generated {hours} hours of temperature data. Average: {np.mean(temps):.1f}°C, Range: {np.min(temps):.1f}°C to {np.max(temps):.1f}°C"
        })
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 Smart Assistant")
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about IoT data, ROI, or valuation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("🧠 Analyzing..."):
            response = ai.analyze(prompt)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

with col2:
    st.header("📈 Live Dashboard")
    
    # Simulated live data
    current_time = datetime.now().strftime("%H:%M:%S")
    current_temp = 22.5 + np.random.randn()
    
    st.metric("🕒 Time", current_time)
    st.metric("🌡️ Temp", f"{current_temp:.1f}°C")
    st.metric("💧 Humidity", f"{45 + np.random.randn()*5:.1f}%")
    st.metric("🔋 Battery", "94%")
    
    # Quick stats
    st.header("📊 Statistics")
    st.metric("Chats", len(st.session_state.messages)//2)
    st.metric("AI Version", "Local v2.1")
    st.metric("Processing", "Instant")
    
    # Actions
    st.header("⚡ Actions")
    if st.button("🔄 Refresh"):
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Free IoT Assistant**")
st.sidebar.markdown("No API keys | Local processing")
st.sidebar.markdown(f"Updated: {datetime.now().strftime('%H:%M')}")

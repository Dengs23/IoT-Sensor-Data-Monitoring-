import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import streamlit as st

# Load environment variables for API keys
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

def calculate_roi(building_data: Dict[str, Any], upgrade_costs: float, energy_prices: float) -> Dict[str, Any]:
    """Calculate ROI for HVAC upgrades and IoT monitoring system"""
    current_energy_usage = estimate_energy_usage(building_data)
    current_energy_cost = current_energy_usage * energy_prices
    
    efficiency_improvement = building_data.get('efficiency_improvement', 0.3)
    upgraded_energy_cost = current_energy_cost * (1 - efficiency_improvement)
    
    annual_savings = current_energy_cost - upgraded_energy_cost
    payback_period = upgrade_costs / annual_savings if annual_savings > 0 else float('inf')
    
    base_value = building_data.get('property_value', 1000000)
    risk_reduction = building_data.get('risk_reduction_score', 0.1)
    efficiency_gain = building_data.get('efficiency_improvement', 0.15)
    value_increase_percent = (risk_reduction * 0.5 + efficiency_gain * 0.5) * 100
    value_increase = base_value * (value_increase_percent / 100)
    
    roi_10_year = ((annual_savings * 10) + value_increase - upgrade_costs) / upgrade_costs
    
    return {
        'current_energy_cost': round(current_energy_cost, 2),
        'upgraded_energy_cost': round(upgraded_energy_cost, 2),
        'annual_savings': round(annual_savings, 2),
        'payback_years': round(payback_period, 1),
        'property_value_increase': round(value_increase, 2),
        '10_year_roi': round(roi_10_year * 100, 1),
    }

# IoT Data Tools for LangChain
class IoTDataTools:
    """Collection of tools for IoT data analysis"""
    
    @staticmethod
    def get_sensor_data(hours: int = 24) -> pd.DataFrame:
        """Generate simulated sensor data"""
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        
        data = {
            'timestamp': timestamps[::-1],
            'temperature': np.random.normal(22, 3, hours).tolist(),
            'humidity': np.random.normal(45, 10, hours).tolist(),
            'battery': np.random.uniform(85, 100, hours).tolist(),
            'location': ['Building_A'] * hours
        }
        
        # Add anomalies
        anomaly_indices = np.random.choice(hours, size=2, replace=False)
        for idx in anomaly_indices:
            data['temperature'][idx] += np.random.choice([-8, 10])
        
        return pd.DataFrame(data)
    
    @staticmethod
    def detect_anomalies(temperature_data: List[float]) -> Dict[str, Any]:
        """Detect temperature anomalies"""
        mean_temp = np.mean(temperature_data)
        std_temp = np.std(temperature_data)
        anomalies = []
        
        for temp in temperature_data:
            if abs(temp - mean_temp) > 2 * std_temp:
                anomalies.append(1)
            else:
                anomalies.append(0)
        
        anomaly_count = sum(anomalies)
        risk_score = min(1.0, anomaly_count / len(temperature_data))
        
        return {
            'anomaly_count': anomaly_count,
            'risk_score': risk_score,
            'mean_temperature': round(mean_temp, 2),
            'temperature_std': round(std_temp, 2)
        }
    
    @staticmethod
    def calculate_roi_for_building(**kwargs) -> Dict[str, Any]:
        """Calculate ROI based on building parameters"""
        building_data = {
            'square_meters': kwargs.get('square_meters', 5000),
            'age_years': kwargs.get('age_years', 15),
            'insulation_rating': kwargs.get('insulation_rating', 6),
            'temperature_variance': kwargs.get('temperature_variance', 0.8),
            'risk_reduction_score': kwargs.get('risk_reduction_score', 0.25),
            'efficiency_improvement': kwargs.get('efficiency_improvement', 0.35),
            'property_value': kwargs.get('property_value', 2500000)
        }
        
        upgrade_costs = kwargs.get('upgrade_costs', 75000)
        energy_prices = kwargs.get('energy_prices', 0.15)
        
        return calculate_roi(building_data, upgrade_costs, energy_prices)

# Create LangChain Tools
def create_iot_tools():
    """Create LangChain tools for IoT data analysis"""
    
    tools = [
        Tool(
            name="GetSensorData",
            func=lambda hours: IoTDataTools.get_sensor_data(int(hours)).to_dict(),
            description="""Get simulated IoT sensor data for analysis. 
            Input: number of hours to get data for (default: 24).
            Output: temperature, humidity, battery data with timestamps."""
        ),
        Tool(
            name="DetectAnomalies",
            func=lambda temp_list: IoTDataTools.detect_anomalies(eval(temp_list)),
            description="""Detect anomalies in temperature data.
            Input: list of temperature values as string (e.g., '[22.5, 23.1, 19.8]').
            Output: anomaly count, risk score, and statistics."""
        ),
        Tool(
            name="CalculateROI",
            func=lambda **kwargs: IoTDataTools.calculate_roi_for_building(**eval(kwargs)),
            description="""Calculate ROI for IoT implementation.
            Input: JSON string with parameters: square_meters, age_years, insulation_rating, 
                   temperature_variance, risk_reduction_score, efficiency_improvement, 
                   property_value, upgrade_costs, energy_prices.
            Example: '{"square_meters": 5000, "upgrade_costs": 75000, "energy_prices": 0.15}'
            Output: ROI analysis results."""
        ),
        Tool(
            name="EstimateBuildingValue",
            func=lambda base_value, risk_score, improvements: 
                f"Estimated value: ${float(base_value) * (1 - float(risk_score) * 0.2) * (1 + float(improvements) * 0.15):,.0f}",
            description="""Estimate building value based on risk and improvements.
            Input: base_value (number), risk_score (0-1), improvements (0-1).
            Output: Estimated building value."""
        )
    ]
    
    return tools

# Streamlit Interface for LangChain Assistant
def create_streamlit_interface():
    """Create Streamlit interface for the LangChain assistant"""
    
    st.set_page_config(
        page_title="🤖 IoT AI Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 IoT Data & ROI AI Assistant")
    st.markdown("Ask questions about your IoT data, get ROI calculations, and receive AI-powered insights")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="""You are an IoT Data Analysis Assistant. You help users understand 
                         their sensor data, calculate ROI for IoT implementations, and provide insights 
                         about building valuation based on temperature readings and anomalies. 
                         You have access to tools for getting sensor data, detecting anomalies, 
                         calculating ROI, and estimating building values. Always be helpful and concise.""")
        ]
    
    # Initialize LangChain agent
    if "agent" not in st.session_state:
        # Initialize LLM (using OpenAI - you'll need an API key)
        try:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.3,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except:
            # Fallback to a mock LLM if no API key
            from langchain.llms.fake import FakeListLLM
            llm = FakeListLLM(responses=[
                "I can help you analyze IoT data and calculate ROI.",
                "Here's an ROI analysis for your building...",
                "The sensor data shows some anomalies."
            ])
        
        # Create tools and agent
        tools = create_iot_tools()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an IoT Data Analysis Assistant. You help with:
             1. Analyzing IoT sensor data (temperature, humidity, battery)
             2. Detecting anomalies and assessing risks
             3. Calculating ROI for IoT implementations
             4. Estimating building values based on sensor data
             
             Use the available tools when you need specific data or calculations.
             Be concise and helpful in your responses.""")
        ])
        
        st.session_state.agent = llm
    
    # Chat interface
    with st.sidebar:
        st.header("💬 Chat with AI Assistant")
        
        # Example questions
        st.subheader("Try asking:")
        example_questions = [
            "Get the latest 48 hours of sensor data",
            "Detect anomalies in my temperature readings",
            "Calculate ROI for a 5000 m² building with $75k upgrade cost",
            "Estimate building value with 25% risk reduction",
            "What's the payback period for IoT sensors?",
            "Show me risk analysis for temperature anomalies"
        ]
        
        for q in example_questions:
            if st.button(q, key=f"q_{hash(q)}"):
                st.session_state.messages.append(HumanMessage(content=q))
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages[1:]:  # Skip system message
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        # Chat input
        if prompt := st.chat_input("Ask about your IoT data or ROI calculations..."):
            # Add user message
            st.session_state.messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Simple response generation (you'd integrate full agent here)
                    if "sensor" in prompt.lower() or "data" in prompt.lower():
                        response = "I can fetch sensor data for you. Typically, we see temperature readings around 22°C with occasional anomalies. Would you like me to get the latest 24 hours of data?"
                    elif "roi" in prompt.lower() or "return" in prompt.lower():
                        response = "Based on typical scenarios, IoT implementations show 3-5 year payback periods and 25-40% ROI over 10 years. I can calculate specific numbers if you provide building details."
                    elif "anomal" in prompt.lower() or "risk" in prompt.lower():
                        response = "Anomaly detection helps identify temperature spikes/drops that could indicate equipment issues. This reduces risk by 20-30% and can prevent costly downtime."
                    elif "value" in prompt.lower() or "valuation" in prompt.lower():
                        response = "Buildings with IoT monitoring typically see 5-15% value increases due to reduced risk and improved efficiency. I can estimate specific values with your building details."
                    else:
                        response = "I can help you with IoT data analysis, ROI calculations, anomaly detection, and building valuation. What specific information would you like?"
                    
                    st.write(response)
                    st.session_state.messages.append(AIMessage(content=response))
    
    # Data visualization section
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Quick Tools")
    
    if st.sidebar.button("📈 Generate Sample Sensor Data"):
        data = IoTDataTools.get_sensor_data(24)
        st.sidebar.dataframe(data.head())
        st.sidebar.success(f"Generated {len(data)} data points")
    
    if st.sidebar.button("💰 Quick ROI Calculator"):
        with st.sidebar.expander("ROI Calculator", expanded=True):
            size = st.number_input("Building Size (m²)", 1000, 10000, 5000)
            cost = st.number_input("Upgrade Cost ($)", 10000, 200000, 75000)
            
            if st.button("Calculate", key="quick_roi"):
                result = IoTDataTools.calculate_roi_for_building(
                    square_meters=size,
                    upgrade_costs=cost
                )
                st.sidebar.metric("10-Year ROI", f"{result['10_year_roi']}%")
                st.sidebar.metric("Payback", f"{result['payback_years']} years")
    
    # API Key setup (for production)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 Setup (Optional)")
    
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("API key set!")

# Main execution
if __name__ == "__main__":
    create_streamlit_interface()

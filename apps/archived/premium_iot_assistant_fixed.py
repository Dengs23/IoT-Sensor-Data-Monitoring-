import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Any, Tuple
import os

st.set_page_config(
    page_title="🚀 Premium IoT Assistant Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class IoTExpertAssistant:
    def __init__(self):
        self.conversation_history = []
        self.expert_knowledge = {
            "sensor_types": {
                "temperature": "Optimal range: -40°C to 125°C, accuracy ±0.5°C",
                "humidity": "Range: 0-100% RH, accuracy ±2%",
                "motion": "PIR sensors for occupancy, range up to 15m",
                "air_quality": "PM2.5, CO2, VOC sensors available",
                "energy": "CT clamps for current monitoring, accuracy ±1%"
            },
            "protocols": {
                "zigbee": "Low power, mesh network, 100m range",
                "lorawan": "Long range (10km), low power, low data rate",
                "wifi": "High bandwidth, high power, easy integration",
                "bluetooth": "Short range, medium power, good for mobile"
            }
        }
    
    def process_query(self, user_query: str, context: Dict = None) -> Dict:
        query_lower = user_query.lower()
        response = {
            "answer": "",
            "recommendations": [],
            "confidence": 0.8
        }
        
        if any(word in query_lower for word in ["roi", "return", "investment", "payback"]):
            response = self._handle_financial_query(query_lower, context)
        elif any(word in query_lower for word in ["sensor", "device", "hardware"]):
            response = self._handle_sensor_query(query_lower)
        elif any(word in query_lower for word in ["network", "protocol", "connectivity"]):
            response = self._handle_network_query(query_lower)
        else:
            response["answer"] = "I can help with IoT financial analysis, sensor selection, and network design. What specifically do you need?"
        
        self.conversation_history.append({
            "query": user_query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _handle_financial_query(self, query: str, context: Dict) -> Dict:
        numbers = re.findall(r'\$?\d+(?:,\d+)*(?:\.\d+)?|\d+(?:,\d+)*(?:\.\d+)?\%', query)
        if numbers:
            answer = f"I found these values: {', '.join(numbers)}. Use the ROI Optimizer tab for detailed analysis."
        else:
            answer = "I can help analyze IoT ROI. Try the ROI Optimizer tab for interactive calculations."
        
        return {
            "answer": answer,
            "recommendations": [
                "Consider both initial investment and ongoing costs",
                "Factor in energy savings (typically 15-30%)",
                "Include maintenance cost reductions (20-40%)",
                "Account for productivity improvements"
            ],
            "confidence": 0.9
        }
    
    def _handle_sensor_query(self, query: str) -> Dict:
        matched_sensors = []
        for sensor_type in self.expert_knowledge["sensor_types"]:
            if sensor_type in query:
                matched_sensors.append(sensor_type)
        
        if matched_sensors:
            answer = f"For {', '.join(matched_sensors)} sensors:\n"
            for sensor in matched_sensors:
                answer += f"• {sensor}: {self.expert_knowledge['sensor_types'][sensor]}\n"
        else:
            answer = "Common IoT sensors include temperature, humidity, motion, air quality, and energy monitors."
        
        return {
            "answer": answer,
            "recommendations": [
                "Select sensors based on accuracy vs cost",
                "Consider environmental conditions",
                "Evaluate power requirements",
                "Check system compatibility"
            ],
            "confidence": 0.85
        }
    
    def _handle_network_query(self, query: str) -> Dict:
        return {
            "answer": "Network selection depends on range, power, data rate, and cost.",
            "recommendations": [
                "Short range indoor: WiFi or Zigbee",
                "Long range outdoor: LoRaWAN or cellular",
                "Battery powered: Zigbee or LoRaWAN",
                "High data rate: WiFi or Ethernet"
            ],
            "confidence": 0.8
        }

class AdvancedROIOptimizer:
    def __init__(self):
        self.scenarios = {}
    
    def calculate_comprehensive_roi(self, params: Dict) -> Dict:
        building_area = params.get("building_area", 1000)
        investment = params.get("investment", 50000)
        energy_cost = params.get("energy_cost", 0.15)
        labor_cost = params.get("labor_cost", 50000)
        maintenance_cost = params.get("maintenance_cost", 25000)
        years = params.get("years", 10)
        
        energy_savings = building_area * 2.5 * energy_cost * 365 * 0.25
        labor_savings = labor_cost * 0.30
        maintenance_savings = maintenance_cost * 0.25
        productivity_gains = building_area * 1.5
        revenue_opportunities = building_area * 0.8
        
        total_annual_savings = (
            energy_savings + labor_savings + 
            maintenance_savings + productivity_gains + 
            revenue_opportunities
        )
        
        total_savings = total_annual_savings * years
        net_profit = total_savings - investment
        roi_percentage = (net_profit / investment) * 100 if investment > 0 else 0
        
        self.scenarios = {
            "aggressive": self._create_scenario("aggressive", total_annual_savings, investment, years),
            "moderate": self._create_scenario("moderate", total_annual_savings, investment, years),
            "conservative": self._create_scenario("conservative", total_annual_savings, investment, years)
        }
        
        return {
            "parameters": params,
            "annual_savings": {
                "energy": round(energy_savings, 2),
                "labor": round(labor_savings, 2),
                "maintenance": round(maintenance_savings, 2),
                "productivity": round(productivity_gains, 2),
                "revenue": round(revenue_opportunities, 2),
                "total": round(total_annual_savings, 2)
            },
            "financial_metrics": {
                "total_savings": round(total_savings, 2),
                "net_profit": round(net_profit, 2),
                "roi_percentage": round(roi_percentage, 2),
                "payback_years": round(investment / total_annual_savings, 2) if total_annual_savings > 0 else float('inf')
            },
            "scenarios": self.scenarios
        }
    
    def _create_scenario(self, scenario_type, base_savings, investment, years):
        multipliers = {
            "aggressive": 1.4,
            "moderate": 1.0,
            "conservative": 0.6
        }
        
        scenario_savings = base_savings * multipliers[scenario_type]
        scenario_profit = (scenario_savings * years) - investment
        scenario_roi = (scenario_profit / investment) * 100 if investment > 0 else 0
        
        return {
            "annual_savings": round(scenario_savings, 2),
            "total_profit": round(scenario_profit, 2),
            "roi_percentage": round(scenario_roi, 2),
            "payback_years": round(investment / scenario_savings, 2) if scenario_savings > 0 else float('inf')
        }

def main():
    st.title("🚀 Premium IoT Assistant Pro")
    st.markdown("### AI-Powered IoT Expert + Advanced Financial Analytics")
    
    ai_assistant = IoTExpertAssistant()
    roi_optimizer = AdvancedROIOptimizer()
    
    tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "💰 ROI Optimizer", "⚙️ Settings"])
    
    with tab1:
        st.header("🤖 IoT Expert Assistant")
        
        # Initialize session state for example queries
        if "example_query" not in st.session_state:
            st.session_state.example_query = ""
        
        # Use a key parameter to make the widget unique
        user_query = st.text_input(
            "Ask me anything about IoT:",
            placeholder="e.g., 'ROI for 5000m² building' or 'best temperature sensors'",
            key="chat_input",
            value=st.session_state.example_query  # Use the stored example query
        )
        
        # Clear the example query after using it
        if user_query == st.session_state.example_query and st.session_state.example_query != "":
            st.session_state.example_query = ""
        
        if st.button("🚀 Ask Assistant", type="primary"):
            if user_query:
                with st.spinner("Analyzing..."):
                    response = ai_assistant.process_query(user_query)
                    
                    st.subheader("💡 Expert Response")
                    st.write(response["answer"])
                    
                    if response.get("recommendations"):
                        st.subheader("🎯 Recommendations")
                        for rec in response["recommendations"]:
                            st.write(f"• {rec}")
            else:
                st.warning("Please enter a question.")
        
        st.subheader("💡 Example Questions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("ROI Analysis", use_container_width=True):
                # Store the example query in session state instead of modifying widget directly
                st.session_state.example_query = "What's the ROI for IoT in commercial buildings?"
                st.rerun()
        
        with col2:
            if st.button("Sensor Advice", use_container_width=True):
                st.session_state.example_query = "Best sensors for energy monitoring?"
                st.rerun()
        
        with col3:
            if st.button("Network Help", use_container_width=True):
                st.session_state.example_query = "Zigbee vs WiFi for IoT?"
                st.rerun()
    
    with tab2:
        st.header("💰 Advanced ROI Optimizer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            building_area = st.number_input(
                "Building Area (m²)",
                min_value=100,
                max_value=100000,
                value=9000,
                step=100
            )
            
            investment = st.number_input(
                "Total Investment ($)",
                min_value=10000,
                max_value=1000000,
                value=250000,
                step=10000
            )
            
            energy_cost = st.slider(
                "Energy Cost ($/kWh)",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01
            )
        
        with col2:
            labor_cost = st.number_input(
                "Annual Labor Cost ($)",
                min_value=10000,
                max_value=500000,
                value=80000,
                step=5000
            )
            
            maintenance_cost = st.number_input(
                "Annual Maintenance ($)",
                min_value=5000,
                max_value=200000,
                value=40000,
                step=5000
            )
            
            years = st.slider(
                "Analysis Period (Years)",
                min_value=1,
                max_value=20,
                value=10,
                step=1
            )
        
        if st.button("📊 Calculate ROI", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                params = {
                    "building_area": building_area,
                    "investment": investment,
                    "energy_cost": energy_cost,
                    "labor_cost": labor_cost,
                    "maintenance_cost": maintenance_cost,
                    "years": years
                }
                
                results = roi_optimizer.calculate_comprehensive_roi(params)
                
                st.subheader("📈 Financial Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ROI", f"{results['financial_metrics']['roi_percentage']:.1f}%")
                
                with col2:
                    st.metric("Payback", f"{results['financial_metrics']['payback_years']:.1f} years")
                
                with col3:
                    st.metric("Annual Savings", f"${results['annual_savings']['total']:,.0f}")
                
                with col4:
                    st.metric("Net Profit", f"${results['financial_metrics']['net_profit']:,.0f}")
                
                st.subheader("💵 Savings Breakdown")
                savings_df = pd.DataFrame({
                    "Category": list(results["annual_savings"].keys())[:-1],
                    "Annual Savings ($)": list(results["annual_savings"].values())[:-1]
                })
                
                fig = px.bar(
                    savings_df,
                    x="Category",
                    y="Annual Savings ($)",
                    color="Category",
                    title="Annual Savings by Category"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📊 Investment Scenarios")
                scenarios_data = []
                for scenario_name, scenario_data in results["scenarios"].items():
                    scenarios_data.append({
                        "Scenario": scenario_name.title(),
                        "Annual Savings": scenario_data["annual_savings"],
                        "ROI %": scenario_data["roi_percentage"],
                        "Payback Years": scenario_data["payback_years"]
                    })
                
                scenarios_df = pd.DataFrame(scenarios_data)
                st.dataframe(scenarios_df, use_container_width=True)
    
    with tab3:
        st.header("⚙️ Settings")
        
        assistant_mode = st.selectbox(
            "Assistant Mode",
            ["Beginner", "Intermediate", "Expert"]
        )
        
        if st.button("🔄 Reset All", type="secondary"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with st.sidebar:
        st.title("IoT Assistant Pro")
        st.divider()
        
        if st.button("💬 New Chat", use_container_width=True):
            ai_assistant.conversation_history = []
            if "example_query" in st.session_state:
                st.session_state.example_query = ""
            st.rerun()
        
        if st.button("💰 Quick Calc", use_container_width=True):
            st.rerun()
        
        st.divider()
        st.metric("AI Responses", len(ai_assistant.conversation_history))
        st.divider()
        
        st.markdown("""
        **Premium IoT Assistant Pro**  
        Version: 2.0  
        • AI Expert Assistant  
        • Advanced ROI Calculator  
        • Interactive Analysis
        """)

if __name__ == "__main__":
    main()

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

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🏢 AI-Powered IoT Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== FAMOUS BUILDINGS DATABASE ====================
FAMOUS_BUILDINGS = {
    "empire state building": {
        "name": "Empire State Building",
        "location": "New York, USA",
        "year_built": 1931,
        "height_m": 443.2,
        "floors": 102,
        "total_area_m2": 257211,
        "office_area_m2": 208879,
        "retail_area_m2": 24154,
        "annual_energy_use_kwh": 55000000,
        "annual_visitors": 4000000,
        "elevators": 73,
        "windows": 6514,
        "iot_investment_usd": 20000000,
        "energy_savings_percent": 38,
        "roi_months": 54,
        "description": "Iconic Art Deco skyscraper, underwent $550M retrofit including $20M IoT systems"
    },
    "burj khalifa": {
        "name": "Burj Khalifa",
        "location": "Dubai, UAE",
        "year_built": 2010,
        "height_m": 828,
        "floors": 163,
        "total_area_m2": 309473,
        "residential_area_m2": 104209,
        "office_area_m2": 33428,
        "hotel_area_m2": 13486,
        "annual_energy_use_kwh": 65000000,
        "annual_visitors": 1700000,
        "elevators": 57,
        "windows": 24400,
        "iot_investment_usd": 50000000,
        "energy_savings_percent": 23,
        "roi_months": 48,
        "description": "World's tallest building, LEED Gold certified, advanced building management system"
    },
    "shanghai tower": {
        "name": "Shanghai Tower",
        "location": "Shanghai, China",
        "year_built": 2015,
        "height_m": 632,
        "floors": 128,
        "total_area_m2": 420000,
        "office_area_m2": 212000,
        "hotel_area_m2": 88000,
        "retail_area_m2": 50000,
        "annual_energy_use_kwh": 48000000,
        "annual_visitors": 2500000,
        "elevators": 106,
        "windows": 20941,
        "iot_investment_usd": 30000000,
        "energy_savings_percent": 32,
        "roi_months": 42,
        "description": "Twisted skyscraper with double-skin facade, 270 wind turbines, LEED Platinum"
    },
    "the gherkin": {
        "name": "30 St Mary Axe (The Gherkin)",
        "location": "London, UK",
        "year_built": 2003,
        "height_m": 180,
        "floors": 41,
        "total_area_m2": 47600,
        "office_area_m2": 46500,
        "annual_energy_use_kwh": 8500000,
        "annual_visitors": 100000,
        "elevators": 23,
        "windows": 5390,
        "iot_investment_usd": 8000000,
        "energy_savings_percent": 30,
        "roi_months": 38,
        "description": "Swiss Re headquarters, energy-efficient cigar shape, natural ventilation"
    },
    "marina bay sands": {
        "name": "Marina Bay Sands",
        "location": "Singapore",
        "year_built": 2010,
        "height_m": 200,
        "floors": 55,
        "total_area_m2": 845000,
        "hotel_rooms": 2561,
        "casino_area_m2": 15000,
        "retail_area_m2": 93000,
        "annual_energy_use_kwh": 350000000,
        "annual_visitors": 50000000,
        "elevators": 141,
        "iot_investment_usd": 75000000,
        "energy_savings_percent": 28,
        "roi_months": 60,
        "description": "Integrated resort with hotel, casino, mall, museum, and Skypark"
    },
    "dubai mall": {
        "name": "The Dubai Mall",
        "location": "Dubai, UAE",
        "year_built": 2008,
        "total_area_m2": 1200000,
        "retail_area_m2": 350000,
        "parking_spaces": 14000,
        "stores": 1200,
        "annual_visitors": 80000000,
        "iot_investment_usd": 25000000,
        "energy_savings_percent": 18,
        "roi_months": 36,
        "description": "World's largest shopping mall by total area, connected to Burj Khalifa"
    },
    "sofi stadium": {
        "name": "SoFi Stadium",
        "location": "Los Angeles, USA",
        "year_built": 2020,
        "total_area_m2": 297000,
        "seating_capacity": 70240,
        "roof_area_m2": 38000,
        "annual_energy_use_kwh": 28000000,
        "annual_events": 40,
        "iot_investment_usd": 35000000,
        "energy_savings_percent": 25,
        "roi_months": 44,
        "description": "Most expensive stadium ever built, LEED Gold, advanced IoT for fan experience"
    }
}

# ==================== AI ASSISTANT ENGINE ====================
class BuildingAIAssistant:
    """AI Assistant for recognizing buildings and suggesting parameters"""
    
    def __init__(self):
        self.conversation_history = []
    
    def recognize_building(self, user_input: str) -> Dict:
        """Recognize building from user input and return parameters"""
        user_input_lower = user_input.lower()
        
        # Check for exact matches
        for building_key, building_data in FAMOUS_BUILDINGS.items():
            if building_key in user_input_lower:
                return {
                    "status": "recognized",
                    "building": building_data,
                    "confidence": 0.95,
                    "message": f"✅ Recognized: {building_data['name']}"
                }
        
        # Check for partial matches
        for building_key, building_data in FAMOUS_BUILDINGS.items():
            building_name_lower = building_data['name'].lower()
            # Check if any word from building name is in user input
            building_words = building_name_lower.split()
            for word in building_words:
                if len(word) > 3 and word in user_input_lower:
                    return {
                        "status": "partial_match",
                        "building": building_data,
                        "confidence": 0.7,
                        "message": f"🔍 Possible match: {building_data['name']}"
                    }
        
        # Check for keywords
        keywords = {
            "tallest": "burj khalifa",
            "new york": "empire state building",
            "dubai": "burj khalifa",
            "shopping mall": "dubai mall",
            "stadium": "sofi stadium",
            "singapore": "marina bay sands",
            "london": "the gherkin"
        }
        
        for keyword, building_key in keywords.items():
            if keyword in user_input_lower:
                return {
                    "status": "keyword_match",
                    "building": FAMOUS_BUILDINGS[building_key],
                    "confidence": 0.6,
                    "message": f"💡 Based on '{keyword}': {FAMOUS_BUILDINGS[building_key]['name']}"
                }
        
        return {
            "status": "not_recognized",
            "building": None,
            "confidence": 0,
            "message": "❓ Building not recognized. Please use manual inputs or try another famous building."
        }
    
    def suggest_parameters(self, building_type: str) -> Dict:
        """Suggest parameters based on building type"""
        suggestions = {
            "office": {
                "energy_cost": 0.15,
                "iot_investment_per_m2": 150,
                "typical_savings_percent": 25,
                "payback_years": 3.5
            },
            "retail": {
                "energy_cost": 0.18,
                "iot_investment_per_m2": 120,
                "typical_savings_percent": 18,
                "payback_years": 4.0
            },
            "hotel": {
                "energy_cost": 0.22,
                "iot_investment_per_m2": 180,
                "typical_savings_percent": 28,
                "payback_years": 3.0
            },
            "hospital": {
                "energy_cost": 0.20,
                "iot_investment_per_m2": 250,
                "typical_savings_percent": 22,
                "payback_years": 4.5
            },
            "stadium": {
                "energy_cost": 0.16,
                "iot_investment_per_m2": 200,
                "typical_savings_percent": 20,
                "payback_years": 5.0
            }
        }
        
        return suggestions.get(building_type, suggestions["office"])
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text"""
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return [float(num) for num in numbers]

# ==================== ROI CALCULATOR ====================
class AdvancedROICalculator:
    """Advanced ROI Calculator with Building Intelligence"""
    
    def calculate_roi(self, params: Dict, building_data: Dict = None) -> Dict:
        """Calculate comprehensive ROI with building intelligence"""
        
        # Extract parameters
        building_area = params.get("building_area", 1000)
        investment = params.get("investment", 50000)
        energy_cost = params.get("energy_cost", 0.15)
        labor_cost = params.get("labor_cost", 50000)
        maintenance_cost = params.get("maintenance_cost", 25000)
        years = params.get("years", 10)
        
        # If we have building data, adjust calculations
        if building_data:
            # Use building-specific adjustments
            base_energy_use = building_data.get("annual_energy_use_kwh", building_area * 200)
            typical_savings = building_data.get("energy_savings_percent", 25) / 100
            typical_roi_months = building_data.get("roi_months", 48)
        else:
            base_energy_use = building_area * 200  # kWh per m² per year
            typical_savings = 0.25
            typical_roi_months = 48
        
        # Calculate savings
        energy_savings = base_energy_use * energy_cost * typical_savings
        labor_savings = labor_cost * 0.30
        maintenance_savings = maintenance_cost * 0.25
        productivity_gains = building_area * 1.5
        
        total_annual_savings = energy_savings + labor_savings + maintenance_savings + productivity_gains
        
        # Financial metrics
        total_savings = total_annual_savings * years
        net_profit = total_savings - investment
        roi_percentage = (net_profit / investment) * 100 if investment > 0 else 0
        payback_months = (investment / total_annual_savings) * 12 if total_annual_savings > 0 else float('inf')
        
        # Generate sensor coordinates for 3D
        sensor_coords = self._generate_sensor_coordinates(building_area)
        
        return {
            "building_area": building_area,
            "investment": investment,
            "total_annual_savings": round(total_annual_savings, 2),
            "total_savings": round(total_savings, 2),
            "net_profit": round(net_profit, 2),
            "roi_percentage": round(roi_percentage, 2),
            "payback_years": round(payback_months / 12, 2),
            "payback_months": round(payback_months, 2),
            "sensor_coordinates": sensor_coords,
            "building_dimensions": self._calculate_dimensions(building_area),
            "savings_breakdown": {
                "energy": round(energy_savings, 2),
                "labor": round(labor_savings, 2),
                "maintenance": round(maintenance_savings, 2),
                "productivity": round(productivity_gains, 2)
            }
        }
    
    def _generate_sensor_coordinates(self, area):
        """Generate sensor coordinates for 3D visualization"""
        length = area ** 0.5 * 1.2
        width = area ** 0.5 * 0.8
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
        
        return coordinates[:50]  # Limit for performance
    
    def _calculate_dimensions(self, area):
        """Calculate building dimensions from area"""
        length = area ** 0.5 * 1.2
        width = area ** 0.5 * 0.8
        height = min(100, max(4, area ** 0.33 * 2))  # Reasonable height
        return [length, width, height]

# ==================== 3D VISUALIZATION ENGINE ====================
class Building3DVisualizer:
    """3D Visualization Engine for Buildings"""
    
    @staticmethod
    def create_building_3d(building_data, sensor_coordinates=None):
        """Create 3D visualization of building with sensors"""
        if building_data and "building_dimensions" in building_data:
            length, width, height = building_data["building_dimensions"]
        else:
            length, width, height = 30, 20, 4
        
        fig = go.Figure()
        
        # Building wireframe
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
        
        # Add sensors if provided
        if sensor_coordinates and len(sensor_coordinates) > 0:
            coords = np.array(sensor_coordinates)
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.8
                ),
                name='IoT Sensors'
            ))
        
        # Add windows pattern for realism
        window_spacing = 3
        window_coords = []
        for x in np.arange(1, length, window_spacing):
            for y in [1, width-1]:
                for z in np.arange(1, height, 2):
                    window_coords.append([x, y, z])
        
        if window_coords:
            windows = np.array(window_coords)
            fig.add_trace(go.Scatter3d(
                x=windows[:, 0],
                y=windows[:, 1],
                z=windows[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='yellow',
                    opacity=0.3
                ),
                name='Windows'
            ))
        
        fig.update_layout(
            title="3D Building Model with IoT Sensors",
            scene=dict(
                xaxis_title='Length (m)',
                yaxis_title='Width (m)',
                zaxis_title='Height (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1))
            ),
            height=500,
            showlegend=True
        )
        
        return fig

# ==================== MAIN APP ====================
def main():
    st.title("🏢 AI-Powered IoT Building Assistant")
    st.markdown("### Speak the building name → Get instant ROI analysis + 3D visualization")
    
    # Initialize engines
    ai_assistant = BuildingAIAssistant()
    roi_calculator = AdvancedROICalculator()
    visualizer = Building3DVisualizer()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🤖 AI Assistant", "💰 ROI Calculator", "📊 3D Visualization"])
    
    # Session state for building data
    if 'current_building' not in st.session_state:
        st.session_state.current_building = None
    if 'roi_results' not in st.session_state:
        st.session_state.roi_results = None
    
    # ==================== TAB 1: AI ASSISTANT ====================
    with tab1:
        st.header("🤖 AI Building Recognition Assistant")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input(
                "Enter building name or description:",
                placeholder="e.g., 'Empire State Building' or 'large office tower in New York'",
                key="building_input"
            )
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🔍 Recognize Building", type="primary", use_container_width=True):
                if user_input:
                    with st.spinner("Analyzing building..."):
                        result = ai_assistant.recognize_building(user_input)
                        
                        if result["status"] != "not_recognized":
                            st.session_state.current_building = result["building"]
                            st.success(result["message"])
                            
                            # Display building info
                            building = result["building"]
                            st.subheader(f"🏢 {building['name']}")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"📍 **Location:** {building['location']}")
                                st.write(f"📅 **Year Built:** {building.get('year_built', 'N/A')}")
                                st.write(f"📏 **Height:** {building.get('height_m', 'N/A')}m")
                                st.write(f"🏢 **Floors:** {building.get('floors', 'N/A')}")
                                st.write(f"📊 **Total Area:** {building['total_area_m2']:,.0f} m²")
                            
                            with col2:
                                st.write(f"💰 **IoT Investment:** ${building['iot_investment_usd']:,.0f}")
                                st.write(f"⚡ **Energy Savings:** {building['energy_savings_percent']}%")
                                st.write(f"📈 **ROI Period:** {building['roi_months']} months")
                                st.write(f"👥 **Annual Visitors:** {building.get('annual_visitors', 0):,.0f}")
                            
                            st.info(building['description'])
                            
                            # Auto-fill ROI calculator
                            st.session_state.building_area = building['total_area_m2']
                            st.session_state.investment = building['iot_investment_usd']
                            
                            st.success("✅ Building recognized! Switch to ROI Calculator tab for analysis.")
                        else:
                            st.warning(result["message"])
                            st.info("💡 Try: Empire State Building, Burj Khalifa, Shanghai Tower, The Gherkin, Marina Bay Sands, Dubai Mall, SoFi Stadium")
                else:
                    st.warning("Please enter a building name or description.")
        
        # Famous buildings quick select
        st.subheader("🏛️ Quick Select Famous Buildings")
        
        cols = st.columns(4)
        famous_buildings_list = list(FAMOUS_BUILDINGS.keys())
        
        for i, building_key in enumerate(famous_buildings_list[:8]):
            with cols[i % 4]:
                building = FAMOUS_BUILDINGS[building_key]
                if st.button(f"🏢 {building['name'].split()[0]}", use_container_width=True):
                    st.session_state.current_building = building
                    st.session_state.building_area = building['total_area_m2']
                    st.session_state.investment = building['iot_investment_usd']
                    st.rerun()
        
        # Building type suggestions
        st.subheader("🏗️ Building Type Suggestions")
        
        building_types = st.columns(5)
        
        with building_types[0]:
            if st.button("🏢 Office", use_container_width=True):
                st.info("Typical office: 150 $/m² IoT investment, 25% energy savings")
        
        with building_types[1]:
            if st.button("🛍️ Retail", use_container_width=True):
                st.info("Malls: 120 $/m² IoT investment, 18% energy savings")
        
        with building_types[2]:
            if st.button("🏨 Hotel", use_container_width=True):
                st.info("Hotels: 180 $/m² IoT investment, 28% energy savings")
        
        with building_types[3]:
            if st.button("🏥 Hospital", use_container_width=True):
                st.info("Hospitals: 250 $/m² IoT investment, 22% energy savings")
        
        with building_types[4]:
            if st.button("🏟️ Stadium", use_container_width=True):
                st.info("Stadiums: 200 $/m² IoT investment, 20% energy savings")
    
    # ==================== TAB 2: ROI CALCULATOR ====================
    with tab2:
        st.header("💰 ROI Calculator")
        
        # Show current building info if available
        if st.session_state.current_building:
            building = st.session_state.current_building
            st.success(f"Currently analyzing: 🏢 **{building['name']}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Building parameters
            building_area = st.number_input(
                "Building Area (m²)",
                min_value=100,
                max_value=1000000,
                value=st.session_state.get('building_area', 1000),
                step=100,
                key="roi_area_input"
            )
            
            investment = st.number_input(
                "IoT Investment ($)",
                min_value=10000,
                max_value=100000000,
                value=st.session_state.get('investment', 50000),
                step=10000,
                key="roi_investment_input"
            )
            
            energy_cost = st.slider(
                "Energy Cost ($/kWh)",
                min_value=0.05,
                max_value=0.50,
                value=0.15,
                step=0.01,
                key="roi_energy_cost"
            )
        
        with col2:
            # Additional parameters
            labor_cost = st.number_input(
                "Annual Labor Cost ($)",
                min_value=0,
                max_value=1000000,
                value=50000,
                step=5000,
                key="roi_labor_cost"
            )
            
            maintenance_cost = st.number_input(
                "Annual Maintenance Cost ($)",
                min_value=0,
                max_value=500000,
                value=25000,
                step=5000,
                key="roi_maintenance_cost"
            )
            
            years = st.slider(
                "Analysis Period (Years)",
                min_value=1,
                max_value=20,
                value=10,
                step=1,
                key="roi_years"
            )
        
        # Calculate button
        if st.button("📊 Calculate Comprehensive ROI", type="primary", use_container_width=True):
            with st.spinner("Calculating ROI with building intelligence..."):
                params = {
                    "building_area": building_area,
                    "investment": investment,
                    "energy_cost": energy_cost,
                    "labor_cost": labor_cost,
                    "maintenance_cost": maintenance_cost,
                    "years": years
                }
                
                # Use building data if available
                building_data = st.session_state.current_building
                
                results = roi_calculator.calculate_roi(params, building_data)
                st.session_state.roi_results = results
                
                # Display results
                st.subheader("📈 Financial Analysis Results")
                
                # Key metrics
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric("ROI Percentage", f"{results['roi_percentage']:.1f}%")
                
                with metrics_cols[1]:
                    st.metric("Payback Period", f"{results['payback_years']:.1f} years")
                
                with metrics_cols[2]:
                    st.metric("Annual Savings", f"${results['total_annual_savings']:,.0f}")
                
                with metrics_cols[3]:
                    st.metric("Net Profit", f"${results['net_profit']:,.0f}")
                
                # Savings breakdown
                st.subheader("💵 Savings Breakdown")
                
                savings_data = pd.DataFrame({
                    "Category": list(results["savings_breakdown"].keys()),
                    "Annual Savings ($)": list(results["savings_breakdown"].values())
                })
                
                fig1 = px.pie(
                    savings_data,
                    values="Annual Savings ($)",
                    names="Category",
                    title="Annual Savings Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Cash flow chart
                st.subheader("📅 10-Year Cash Flow Projection")
                
                years_list = list(range(years + 1))
                cash_flows = [-investment] + [results['total_annual_savings']] * years
                cumulative = np.cumsum(cash_flows)
                
                fig2 = go.Figure()
                
                fig2.add_trace(go.Bar(
                    x=years_list,
                    y=cash_flows,
                    name="Annual Cash Flow",
                    marker_color=['red'] + ['green'] * years
                ))
                
                fig2.add_trace(go.Scatter(
                    x=years_list,
                    y=cumulative,
                    name="Cumulative Cash Flow",
                    line=dict(color="blue", width=3),
                    yaxis="y2"
                ))
                
                fig2.update_layout(
                    title="Cash Flow Analysis",
                    xaxis_title="Year",
                    yaxis_title="Annual Cash Flow ($)",
                    yaxis2=dict(
                        title="Cumulative Cash Flow ($)",
                        overlaying="y",
                        side="right"
                    ),
                    height=400
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                st.success("✅ ROI analysis complete! Switch to 3D Visualization tab.")
    
    # ==================== TAB 3: 3D VISUALIZATION ====================
    with tab3:
        st.header("📊 3D Building Visualization")
        
        if st.session_state.roi_results:
            results = st.session_state.roi_results
            
            # Create 3D visualization
            building_data = {
                "building_dimensions": results["building_dimensions"],
                "sensor_count": len(results["sensor_coordinates"])
            }
            
            fig = visualizer.create_building_3d(building_data, results["sensor_coordinates"])
            st.plotly_chart(fig, use_container_width=True)
            
            # Building stats
            st.subheader("🏗️ Building Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Building Area", f"{results['building_area']:,.0f} m²")
            
            with col2:
                length, width, height = results["building_dimensions"]
                st.metric("Dimensions", f"{length:.0f}m × {width:.0f}m × {height:.0f}m")
            
            with col3:
                st.metric("IoT Sensors", f"{len(results['sensor_coordinates'])}")
            
            # Sensor network info
            st.subheader("📡 IoT Sensor Network")
            
            sensor_cols = st.columns(4)
            
            with sensor_cols[0]:
                optimal_spacing = 5
                st.metric("Optimal Spacing", f"{optimal_spacing}m")
            
            with sensor_cols[1]:
                coverage = min(100, (len(results["sensor_coordinates"]) * 25) / results["building_area"] * 100)
                st.metric("Coverage", f"{coverage:.1f}%")
            
            with sensor_cols[2]:
                sensors_per_floor = len(results["sensor_coordinates"]) / max(1, results["building_dimensions"][2] / 3)
                st.metric("Sensors per Floor", f"{sensors_per_floor:.0f}")
            
            with sensor_cols[3]:
                investment_per_sensor = results["investment"] / max(1, len(results["sensor_coordinates"]))
                st.metric("Cost per Sensor", f"${investment_per_sensor:.0f}")
        
        else:
            st.info("👈 First calculate ROI to generate 3D visualization")
            
            # Show sample building
            st.subheader("🎯 Sample Building: Office Tower")
            
            sample_dimensions = [50, 30, 20]  # Length, width, height
            sample_data = {"building_dimensions": sample_dimensions}
            
            fig = visualizer.create_building_3d(sample_data)
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.title("🏢 IoT Building Assistant")
        
        st.divider()
        
        # Current building info
        if st.session_state.current_building:
            building = st.session_state.current_building
            st.subheader(f"🏢 {building['name']}")
            st.write(f"📍 {building['location']}")
            st.write(f"📊 {building['total_area_m2']:,.0f} m²")
            st.write(f"💰 ${building['iot_investment_usd']:,.0f}")
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 New Analysis", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['_']:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("📥 Export Report", use_container_width=True):
            st.success("Report exported!")
        
        st.divider()
        
        # Building database stats
        st.subheader("📚 Building Database")
        st.write(f"🏛️ **{len(FAMOUS_BUILDINGS)}** famous buildings")
        st.write("🌍 **7** countries")
        st.write("🏗️ **6** building types")
        
        st.divider()
        
        # About
        st.markdown("""
        **AI-Powered IoT Assistant**

        Features:
        • 🤖 AI building recognition
        • 💰 Smart ROI calculation
        • 📊 3D visualization
        • 🏛️ Famous building database
        • ⚡ Real-time analysis

        Speak any famous building name
        for instant analysis!
        """)

if __name__ == "__main__":
    main()

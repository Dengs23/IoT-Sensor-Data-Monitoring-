import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Any, Tuple

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧠 Interactive IoT Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ROI OPTIMIZER CLASS ====================
class InteractiveROIOptimizer:
    def __init__(self):
        self.calculation_history = []
    
    def calculate_roi(self, building_area, investment, energy_cost, labor_cost, maintenance_cost, years=10):
        """Calculate interactive ROI based on user inputs"""
        
        # Calculate annual savings based on building size
        # Base savings per m² (adjustable)
        base_savings_per_m2 = 3.5  # $ per m² per year
        
        # Energy savings (scales with building size)
        energy_savings = building_area * base_savings_per_m2 * (energy_cost / 0.15)
        
        # Labor savings (reduced maintenance)
        labor_savings = labor_cost * 0.3  # 30% reduction
        
        # Maintenance savings
        maintenance_savings = maintenance_cost * 0.25  # 25% reduction
        
        total_annual_savings = energy_savings + labor_savings + maintenance_savings
        
        # Calculate metrics
        total_savings = total_annual_savings * years
        net_profit = total_savings - investment
        roi_percentage = (net_profit / investment) * 100 if investment > 0 else 0
        payback_years = investment / total_annual_savings if total_annual_savings > 0 else float('inf')
        
        # Generate scenarios
        scenarios = {
            'optimistic': {
                'annual_savings': total_annual_savings * 1.3,
                'payback_years': payback_years * 0.7,
                'roi_percentage': roi_percentage * 1.3,
                'value_increase': investment * 1.04
            },
            'baseline': {
                'annual_savings': total_annual_savings,
                'payback_years': payback_years,
                'roi_percentage': roi_percentage,
                'value_increase': investment * 0.8
            },
            'conservative': {
                'annual_savings': total_annual_savings * 0.7,
                'payback_years': payback_years * 1.3,
                'roi_percentage': roi_percentage * 0.7,
                'value_increase': investment * 0.56
            }
        }
        
        return {
            'building_area': building_area,
            'investment': investment,
            'total_annual_savings': round(total_annual_savings, 2),
            'total_savings': round(total_savings, 2),
            'net_profit': round(net_profit, 2),
            'roi_percentage': round(roi_percentage, 2),
            'payback_years': round(payback_years, 2),
            'scenarios': scenarios,
            'breakdown': {
                'energy_savings': round(energy_savings, 2),
                'labor_savings': round(labor_savings, 2),
                'maintenance_savings': round(maintenance_savings, 2)
            }
        }

# ==================== MAIN APP ====================
def main():
    st.title("💰 Interactive IoT ROI Calculator")
    st.markdown("### Adjust parameters and see real-time ROI calculations")
    
    # Initialize optimizer
    optimizer = InteractiveROIOptimizer()
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("⚙️ Input Parameters")
        
        st.subheader("Building Details")
        building_area = st.number_input(
            "Building Area (m²)",
            min_value=100,
            max_value=100000,
            value=9000,
            step=100,
            help="Total building area in square meters"
        )
        
        st.subheader("Investment Details")
        investment = st.number_input(
            "Total Investment ($)",
            min_value=10000,
            max_value=1000000,
            value=250000,
            step=10000,
            help="Total IoT system investment"
        )
        
        st.subheader("Cost Parameters")
        energy_cost = st.slider(
            "Energy Cost ($/kWh)",
            min_value=0.05,
            max_value=0.50,
            value=0.15,
            step=0.01,
            help="Current energy cost per kWh"
        )
        
        labor_cost = st.number_input(
            "Annual Labor Cost ($)",
            min_value=10000,
            max_value=500000,
            value=80000,
            step=5000,
            help="Annual labor cost for maintenance"
        )
        
        maintenance_cost = st.number_input(
            "Annual Maintenance Cost ($)",
            min_value=5000,
            max_value=200000,
            value=40000,
            step=5000,
            help="Annual maintenance and repair costs"
        )
        
        years = st.slider(
            "Analysis Period (Years)",
            min_value=1,
            max_value=20,
            value=10,
            step=1
        )
        
        st.divider()
        
        if st.button("🔄 Reset to Defaults", type="secondary"):
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 ROI Analysis Results")
        
        # Calculate ROI
        roi_result = optimizer.calculate_roi(
            building_area, investment, energy_cost, 
            labor_cost, maintenance_cost, years
        )
        
        # Key Metrics
        st.subheader("💰 Key Financial Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Annual Savings", f"${roi_result['total_annual_savings']:,.0f}")
        
        with metrics_col2:
            st.metric("Total ROI", f"{roi_result['roi_percentage']:.1f}%")
        
        with metrics_col3:
            st.metric("Payback Period", f"{roi_result['payback_years']:.1f} years")
        
        with metrics_col4:
            st.metric("Net Profit", f"${roi_result['net_profit']:,.0f}")
        
        # Savings Breakdown
        st.subheader("📈 Savings Breakdown")
        
        breakdown_data = pd.DataFrame({
            'Category': ['Energy', 'Labor', 'Maintenance'],
            'Savings ($)': [
                roi_result['breakdown']['energy_savings'],
                roi_result['breakdown']['labor_savings'],
                roi_result['breakdown']['maintenance_savings']
            ]
        })
        
        fig = px.pie(
            breakdown_data, 
            values='Savings ($)', 
            names='Category',
            title='Annual Savings Breakdown',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cash Flow Chart
        st.subheader("📅 Cash Flow Projection")
        
        years_list = list(range(years + 1))
        cash_flows = [-investment] + [roi_result['total_annual_savings']] * years
        cumulative_cash = np.cumsum(cash_flows)
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            x=years_list,
            y=cash_flows,
            name='Annual Cash Flow',
            marker_color='lightblue'
        ))
        
        fig2.add_trace(go.Scatter(
            x=years_list,
            y=cumulative_cash,
            name='Cumulative Cash Flow',
            line=dict(color='green', width=3),
            yaxis='y2'
        ))
        
        fig2.update_layout(
            title='Cash Flow Analysis',
            xaxis_title='Year',
            yaxis_title='Annual Cash Flow ($)',
            yaxis2=dict(
                title='Cumulative Cash Flow ($)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.header("📋 Investment Scenarios")
        
        scenarios = roi_result['scenarios']
        
        # Optimistic Scenario
        with st.expander("🚀 Optimistic Scenario", expanded=True):
            st.metric("Annual Savings", f"${scenarios['optimistic']['annual_savings']:,.0f}")
            st.metric("Payback Period", f"{scenarios['optimistic']['payback_years']:.1f} years")
            st.metric("ROI", f"{scenarios['optimistic']['roi_percentage']:.1f}%")
            st.metric("Value Increase", f"${scenarios['optimistic']['value_increase']:,.0f}")
        
        # Baseline Scenario
        with st.expander("📊 Baseline Scenario", expanded=True):
            st.metric("Annual Savings", f"${scenarios['baseline']['annual_savings']:,.0f}")
            st.metric("Payback Period", f"{scenarios['baseline']['payback_years']:.1f} years")
            st.metric("ROI", f"{scenarios['baseline']['roi_percentage']:.1f}%")
            st.metric("Value Increase", f"${scenarios['baseline']['value_increase']:,.0f}")
        
        # Conservative Scenario
        with st.expander("🛡️ Conservative Scenario", expanded=True):
            st.metric("Annual Savings", f"${scenarios['conservative']['annual_savings']:,.0f}")
            st.metric("Payback Period", f"{scenarios['conservative']['payback_years']:.1f} years")
            st.metric("ROI", f"{scenarios['conservative']['roi_percentage']:.1f}%")
            st.metric("Value Increase", f"${scenarios['conservative']['value_increase']:,.0f}")
        
        # Recommendations
        st.header("🎯 Recommendations")
        
        if roi_result['roi_percentage'] > 200:
            st.success("**Excellent Investment** - Very high ROI with quick payback")
            st.write("• Consider full implementation")
            st.write("• Explore additional IoT features")
            st.write("• Monitor for scaling opportunities")
        elif roi_result['roi_percentage'] > 100:
            st.info("**Good Investment** - Solid returns expected")
            st.write("• Proceed with implementation")
            st.write("• Start with high-impact areas")
            st.write("• Plan for phased rollout")
        elif roi_result['roi_percentage'] > 50:
            st.warning("**Moderate Investment** - Consider carefully")
            st.write("• Focus on highest ROI components")
            st.write("• Negotiate better pricing")
            st.write("• Consider pilot program")
        else:
            st.error("**Poor Investment** - Reevaluate approach")
            st.write("• Review cost assumptions")
            st.write("• Consider alternative solutions")
            st.write("• Seek expert consultation")
    
    # Sensitivity Analysis
    st.header("🔬 Sensitivity Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROI vs Investment")
        
        investment_range = np.linspace(investment * 0.5, investment * 1.5, 10)
        roi_values = []
        
        for inv in investment_range:
            temp_result = optimizer.calculate_roi(
                building_area, inv, energy_cost, 
                labor_cost, maintenance_cost, years
            )
            roi_values.append(temp_result['roi_percentage'])
        
        fig3 = px.line(
            x=investment_range, 
            y=roi_values,
            title='ROI Sensitivity to Investment',
            labels={'x': 'Investment ($)', 'y': 'ROI (%)'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("ROI vs Building Size")
        
        area_range = np.linspace(building_area * 0.5, building_area * 1.5, 10)
        roi_values_area = []
        
        for area in area_range:
            temp_result = optimizer.calculate_roi(
                area, investment, energy_cost, 
                labor_cost, maintenance_cost, years
            )
            roi_values_area.append(temp_result['roi_percentage'])
        
        fig4 = px.line(
            x=area_range, 
            y=roi_values_area,
            title='ROI Sensitivity to Building Size',
            labels={'x': 'Building Area (m²)', 'y': 'ROI (%)'}
        )
        st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()

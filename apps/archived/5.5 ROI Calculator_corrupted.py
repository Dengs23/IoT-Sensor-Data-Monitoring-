# First, backup the original file
cp "5.5 ROI Calculator.py" "5.5 ROI Calculator_backup.py"

# Create the new file
cat > "5.5 ROI Calculator.py" << 'EOF'
import numpy as np
import pandas as pd

def estimate_energy_usage(building_data):
    """
    Estimate energy usage based on building characteristics
    building_data should be a dictionary with keys:
    - square_meters
    - age_years
    - insulation_rating (1-10)
    - avg_temperature (from sensors)
    - temperature_variance (from ML predictions)
    """
    base_usage = building_data.get('square_meters', 1000) * 0.15  # kWh/m²/month
    
    # Adjust for building age
    age_factor = 1 + (building_data.get('age_years', 10) * 0.01)
    
    # Adjust for insulation (better insulation = less energy)
    insulation = building_data.get('insulation_rating', 5)
    insulation_factor = 1.2 - (insulation * 0.02)
    
    # Adjust for temperature anomalies (from your ML predictions)
    temp_variance = building_data.get('temperature_variance', 0)
    temp_factor = 1 + (temp_variance * 0.1)
    
    return base_usage * age_factor * insulation_factor * temp_factor

def estimate_value_increase(building_data):
    """
    Estimate property value increase from IoT monitoring and upgrades
    Based on risk reduction and efficiency improvements
    """
    base_value = building_data.get('property_value', 1000000)
    
    # Value increase from risk reduction (temperature stability)
    risk_reduction = building_data.get('risk_reduction_score', 0.1)  # From ML predictions
    
    # Value increase from efficiency improvements
    efficiency_gain = building_data.get('efficiency_improvement', 0.15)
    
    # Combined value increase (typically 5-20% for smart buildings)
    value_increase_percent = (risk_reduction * 0.5 + efficiency_gain * 0.5) * 100
    
    return base_value * (value_increase_percent / 100)

def calculate_roi(building_data, upgrade_costs, energy_prices):
    """
    Calculate ROI for HVAC upgrades and IoT monitoring system
    """
    # Estimate current energy usage
    current_energy_usage = estimate_energy_usage(building_data)
    current_energy_cost = current_energy_usage * energy_prices
    
    # Estimate improved energy usage (30% savings from upgrades + IoT optimization)
    efficiency_improvement = building_data.get('efficiency_improvement', 0.3)
    upgraded_energy_cost = current_energy_cost * (1 - efficiency_improvement)
    
    # Calculate savings
    annual_savings = current_energy_cost - upgraded_energy_cost
    payback_period = upgrade_costs / annual_savings if annual_savings > 0 else float('inf')
    
    # Property value increase
    value_increase = estimate_value_increase(building_data)
    
    # ROI calculations
    roi_10_year = ((annual_savings * 10) + value_increase - upgrade_costs) / upgrade_costs
    
    return {
        'current_energy_cost': round(current_energy_cost, 2),
        'upgraded_energy_cost': round(upgraded_energy_cost, 2),
        'annual_savings': round(annual_savings, 2),
        'monthly_savings': round(annual_savings / 12, 2),
        'payback_years': round(payback_period, 1),
        'property_value_increase': round(value_increase, 2),
        'value_increase_percent': round((value_increase / building_data.get('property_value', 1000000)) * 100, 1),
        '10_year_roi': round(roi_10_year * 100, 1),  # As percentage
        'total_10_year_benefit': round((annual_savings * 10) + value_increase, 2),
        'net_10_year_gain': round(((annual_savings * 10) + value_increase - upgrade_costs), 2)
    }

# Example usage function
def example_calculation():
    """Show how to use the ROI calculator with sample data"""
    # Sample building data (could come from your ML predictions)
    building_data = {
        'square_meters': 5000,
        'age_years': 15,
        'insulation_rating': 6,
        'avg_temperature': 22.5,
        'temperature_variance': 0.8,  # From ML anomaly detection
        'risk_reduction_score': 0.25,  # 25% risk reduction from monitoring
        'efficiency_improvement': 0.35,  # 35% efficiency improvement
        'property_value': 2500000
    }
    
    # System costs
    upgrade_costs = 75000  # IoT sensors + HVAC upgrades
    energy_price = 0.15  # $ per kWh
    
    # Calculate ROI
    results = calculate_roi(building_data, upgrade_costs, energy_price)
    
    print("=== ROI Analysis for Industrial Building ===")
    for key, value in results.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return results

if __name__ == "__main__":
    # Run example when file is executed directly
    example_calculation()
EOF



    
    

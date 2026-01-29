def calculate_roi(building_data, upgrade_costs, energy_prices):
    """Calculate ROI for HVAC upgrades"""
    current_energy_cost = estimate_energy_usage(building_data) * energy_prices
    upgraded_energy_cost = current_energy_cost * 0.7  # Assume 30% savings
    annual_savings = current_energy_cost - upgraded_energy_cost
    payback_period = upgrade_costs / annual_savings
    
    return {
        'annual_savings': annual_savings,
        'payback_years': payback_period,
        '10_year_roi': (annual_savings * 10 - upgrade_costs) / upgrade_costs,
        'property_value_increase': estimate_value_increase(building_data)
    }
    
    

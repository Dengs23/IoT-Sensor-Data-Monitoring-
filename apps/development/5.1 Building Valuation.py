# ==================== PHASE 1: ENHANCED SIMULATION ====================
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class BuildingValuationSimulator:
    def __init__(self, n_buildings=10, n_days=365):
        self.n_buildings = n_buildings
        self.n_days = n_days
        self.buildings = self._generate_building_metadata()
        
    def _generate_building_metadata(self):
        """Create realistic building profiles for valuation"""
        building_types = ['Office Building', 'Warehouse', 'Manufacturing Plant', 
                         'Data Center', 'Retail Mall', 'Hospital', 'Hotel']
        
        buildings = []
        for i in range(self.n_buildings):
            btype = np.random.choice(building_types)
            base_value = {
                'Office Building': np.random.uniform(5e6, 50e6),
                'Warehouse': np.random.uniform(2e6, 20e6),
                'Manufacturing Plant': np.random.uniform(10e6, 100e6),
                'Data Center': np.random.uniform(50e6, 500e6),
                'Retail Mall': np.random.uniform(20e6, 200e6),
                'Hospital': np.random.uniform(100e6, 500e6),
                'Hotel': np.random.uniform(15e6, 150e6)
            }[btype]
            
            buildings.append({
                'building_id': f'B{str(i+1).zfill(3)}',
                'building_type': btype,
                'age_years': np.random.randint(1, 50),
                'square_feet': np.random.randint(10000, 500000),
                'location_quality': np.random.uniform(0.5, 1.0),  # 0-1 score
                'energy_certificate': np.random.choice(['A', 'B', 'C', 'D', 'E']),
                'current_occupancy': np.random.uniform(0.5, 0.95),
                'base_value': base_value,
                'temperature_zones': np.random.randint(3, 15)  # Number of sensor zones
            })
        
        return pd.DataFrame(buildings)
    
    def generate_sensor_data(self):
        """Generate temperature data with realistic patterns"""
        all_data = []
        
        for _, building in self.buildings.iterrows():
            building_id = building['building_id']
            n_zones = building['temperature_zones']
            base_temp = {
                'Office Building': 22.0,
                'Warehouse': 18.0,
                'Manufacturing Plant': 20.0,
                'Data Center': 19.0,  # Cooler for servers
                'Retail Mall': 23.0,
                'Hospital': 24.0,
                'Hotel': 23.0
            }[building['building_type']]
            
            for day in range(self.n_days):
                date = datetime(2024, 1, 1) + timedelta(days=day)
                
                # Add seasonal patterns
                seasonal_effect = 5 * np.sin(2 * np.pi * day / 365)
                
                # Add weekday/weekend patterns
                is_weekend = 1 if date.weekday() >= 5 else 0
                occupancy_factor = 0.3 if is_weekend else 1.0
                
                # Add efficiency factor (older buildings less efficient)
                efficiency = 1 - (building['age_years'] / 100)
                
                for zone in range(n_zones):
                    # Base temperature with variations
                    temp = base_temp + seasonal_effect
                    
                    # Zone variations
                    temp += np.random.normal(0, 2)
                    
                    # Occupancy effect
                    temp += building['current_occupancy'] * occupancy_factor * 2
                    
                    # Inefficiency "penalty" for older buildings
                    if building['age_years'] > 20:
                        temp += np.random.uniform(0, 3)  # Less stable temps
                    
                    # Add some anomalies
                    anomaly = np.random.choice([0, 1], p=[0.95, 0.05])
                    if anomaly:
                        temp += np.random.uniform(-5, 5)
                    
                    all_data.append({
                        'timestamp': date,
                        'building_id': building_id,
                        'zone_id': f'{building_id}_Z{zone}',
                        'temperature': round(temp, 2),
                        'day_of_year': day,
                        'is_weekend': is_weekend,
                        'hour': np.random.randint(0, 24)  # Simulate hourly readings
                    })
        
        return pd.DataFrame(all_data)
    
    def calculate_energy_efficiency_score(self, sensor_df):
        """Calculate energy efficiency from temperature stability"""
        efficiency_scores = {}
        
        for building_id in sensor_df['building_id'].unique():
            bdata = sensor_df[sensor_df['building_id'] == building_id]
            building_info = self.buildings[self.buildings['building_id'] == building_id].iloc[0]
            
            # Metrics for efficiency
            temp_std = bdata['temperature'].std()  # Lower std = more stable = more efficient
            temp_range = bdata['temperature'].max() - bdata['temperature'].min()
            
            # Normalize scores
            std_score = max(0, 1 - (temp_std / 10))  # 0-1 scale
            range_score = max(0, 1 - (temp_range / 20))
            
            # Weighted efficiency score
            efficiency_score = (std_score * 0.6 + range_score * 0.4) * 100
            
            # Adjust for building age and type
            age_penalty = max(0, (building_info['age_years'] - 10) * 0.01)
            efficiency_score *= (1 - age_penalty)
            
            efficiency_scores[building_id] = min(100, efficiency_score)
        
        return efficiency_scores

# ==================== PHASE 2: ML FOR VALUATION ====================
class BuildingValuationModel:
    def __init__(self):
        self.models = {
            'efficiency_predictor': xgb.XGBRegressor(),
            'valuation_predictor': RandomForestRegressor(n_estimators=100),
            'maintenance_predictor': RandomForestRegressor(n_estimators=50)
        }
        self.scaler = StandardScaler()
        
    def extract_features(self, sensor_df, building_df, efficiency_scores):
        """Create features for ML models"""
        features = []
        
        for building_id in building_df['building_id'].unique():
            bdata = sensor_df[sensor_df['building_id'] == building_id]
            building_info = building_df[building_df['building_id'] == building_id].iloc[0]
            
            # Temperature-based features
            temp_features = {
                'temp_mean': bdata['temperature'].mean(),
                'temp_std': bdata['temperature'].std(),
                'temp_range': bdata['temperature'].max() - bdata['temperature'].min(),
                'temp_above_25': (bdata['temperature'] > 25).mean(),  # Cooling inefficiency
                'temp_below_18': (bdata['temperature'] < 18).mean(),  # Heating inefficiency
                'temp_stability_score': efficiency_scores.get(building_id, 50),
                'daily_variation': bdata.groupby(bdata['timestamp'].dt.date)['temperature'].std().mean(),
                'zone_variation': bdata.groupby('zone_id')['temperature'].std().mean()
            }
            
            # Combine with building features
            feature_row = {
                'building_id': building_id,
                'age_years': building_info['age_years'],
                'square_feet': building_info['square_feet'],
                'location_quality': building_info['location_quality'],
                'occupancy': building_info['current_occupancy'],
                'n_zones': building_info['temperature_zones'],
                **temp_features,
                'building_type_encoded': hash(building_info['building_type']) % 100  # Simple encoding
            }
            
            features.append(feature_row)
        
        return pd.DataFrame(features)
    
    def predict_valuation(self, features_df, building_df):
        """Predict building value based on temperature efficiency"""
        # Target: deviation from "ideal" value
        building_df['efficiency_multiplier'] = features_df['temp_stability_score'] / 100
        
        # Simulate market value adjustment based on efficiency
        base_values = building_df.set_index('building_id')['base_value']
        efficiency_scores = features_df.set_index('building_id')['temp_stability_score']
        
        # Simple valuation model: efficiency affects value by ±15%
        predicted_values = {}
        for building_id in base_values.index:
            base = base_values[building_id]
            eff_score = efficiency_scores[building_id]
            
            # Efficiency premium/discount
            efficiency_effect = (eff_score - 70) / 100  # 70 as baseline
            
            # Other factors
            age_penalty = max(0, (building_df.loc[building_df['building_id'] == building_id, 'age_years'].iloc[0] - 20) * 0.01)
            occupancy_premium = (building_df.loc[building_df['building_id'] == building_id, 'current_occupancy'].iloc[0] - 0.7) * 0.1
            
            # Final predicted value
            predicted_value = base * (1 + efficiency_effect - age_penalty + occupancy_premium)
            predicted_values[building_id] = predicted_value
        
        return predicted_values
    
    def calculate_investment_metrics(self, predicted_values, building_df):
        """Calculate ROI and investment viability"""
        metrics = []
        
        for building_id, pred_value in predicted_values.items():
            building = building_df[building_df['building_id'] == building_id].iloc[0]
            base_value = building['base_value']
            
            # Price-to-Efficiency Ratio (PER) - lower is better
            per = pred_value / (features_df.loc[features_df['building_id'] == building_id, 'temp_stability_score'].iloc[0])
            
            # Efficiency-Adjusted Cap Rate (simplified)
            # Assume net operating income is 8% of value, adjusted by efficiency
            noi = pred_value * 0.08
            cap_rate = (noi / pred_value) * 100
            
            # Investment score
            efficiency_score = features_df.loc[features_df['building_id'] == building_id, 'temp_stability_score'].iloc[0]
            age = building['age_years']
            
            investment_score = (
                efficiency_score * 0.4 +
                (100 - min(age, 50) * 2) * 0.3 +  # Age penalty
                building['location_quality'] * 100 * 0.2 +
                building['current_occupancy'] * 100 * 0.1
            )
            
            # Recommendation
            if investment_score > 80:
                recommendation = "STRONG BUY"
                reasoning = "High efficiency, good location, stable temperatures"
            elif investment_score > 65:
                recommendation = "BUY"
                reasoning = "Good efficiency profile, reasonable valuation"
            elif investment_score > 50:
                recommendation = "HOLD"
                reasoning = "Average efficiency, consider upgrades"
            else:
                recommendation = "AVOID"
                reasoning = "Poor temperature control indicates system issues"
            
            metrics.append({
                'building_id': building_id,
                'building_type': building['building_type'],
                'base_value': f"${base_value:,.0f}",
                'predicted_value': f"${pred_value:,.0f}",
                'value_adjustment': f"{((pred_value/base_value)-1)*100:+.1f}%",
                'efficiency_score': f"{efficiency_score:.1f}/100",
                'price_to_efficiency': f"{per:,.0f}",
                'cap_rate': f"{cap_rate:.1f}%",
                'investment_score': f"{investment_score:.1f}/100",
                'recommendation': recommendation,
                'reasoning': reasoning
            })
        
        return pd.DataFrame(metrics)

# ==================== PHASE 3: VISUALIZATION & DASHBOARD ====================
class InvestmentDashboard:
    def __init__(self, simulator, model):
        self.simulator = simulator
        self.model = model
        
    def create_dashboard(self):
        # Generate data
        print("Generating building data...")
        building_df = self.simulator.buildings
        sensor_df = self.simulator.generate_sensor_data()
        
        print("Calculating efficiency scores...")
        efficiency_scores = self.simulator.calculate_energy_efficiency_score(sensor_df)
        
        print("Extracting ML features...")
        features_df = self.model.extract_features(sensor_df, building_df, efficiency_scores)
        
        print("Predicting valuations...")
        predicted_values = self.model.predict_valuation(features_df, building_df)
        
        print("Calculating investment metrics...")
        investment_df = self.model.calculate_investment_metrics(predicted_values, building_df)
        
        # Create visual dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Building Efficiency Scores', 'Predicted vs Base Value',
                          'Investment Score by Building', 'Efficiency vs Age',
                          'Temperature Stability', 'Cap Rate Distribution',
                          'Top Investment Opportunities', 'Building Type Analysis',
                          'Investment Recommendations'),
            specs=[[{'type': 'bar'}, {'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'box'}],
                   [{'type': 'table'}, {'type': 'pie'}, {'type': 'heatmap'}]]
        )
        
        # 1. Efficiency scores
        fig.add_trace(
            go.Bar(x=list(efficiency_scores.keys()), 
                   y=list(efficiency_scores.values()),
                   name='Efficiency Score'),
            row=1, col=1
        )
        
        # 2. Predicted vs Base Value
        base_vals = [float(v.replace('$', '').replace(',', '')) 
                    for v in investment_df['base_value']]
        pred_vals = [float(v.replace('$', '').replace(',', '')) 
                    for v in investment_df['predicted_value']]
        
        fig.add_trace(
            go.Scatter(x=base_vals, y=pred_vals, mode='markers',
                      text=investment_df['building_id'],
                      name='Valuation Comparison'),
            row=1, col=2
        )
        
        # Add perfect correlation line
        max_val = max(max(base_vals), max(pred_vals))
        fig.add_trace(
            go.Scatter(x=[0, max_val], y=[0, max_val], 
                      mode='lines', name='Parity Line',
                      line=dict(dash='dash')),
            row=1, col=2
        )
        
        # 3. Investment scores
        fig.add_trace(
            go.Bar(x=investment_df['building_id'], 
                   y=investment_df['investment_score'].str.extract('(\d+\.?\d*)')[0].astype(float),
                   marker_color=['green' if s > 65 else 'yellow' if s > 50 else 'red' 
                                for s in investment_df['investment_score'].str.extract('(\d+\.?\d*)')[0].astype(float)]),
            row=1, col=3
        )
        
        # 4. Efficiency vs Age
        ages = building_df['age_years']
        efficiencies = [efficiency_scores.get(bid, 50) for bid in building_df['building_id']]
        
        fig.add_trace(
            go.Scatter(x=ages, y=efficiencies, mode='markers',
                      text=building_df['building_id'],
                      name='Age vs Efficiency'),
            row=2, col=1
        )
        
        # 5. Temperature stability histogram
        fig.add_trace(
            go.Histogram(x=sensor_df['temperature'], nbinsx=30,
                        name='Temperature Distribution'),
            row=2, col=2
        )
        
        # 6. Cap rate distribution
        cap_rates = investment_df['cap_rate'].str.extract('(\d+\.?\d*)')[0].astype(float)
        fig.add_trace(
            go.Box(y=cap_rates, name='Cap Rates'),
            row=2, col=3
        )
        
        # 7. Top investments table
        top_investments = investment_df.sort_values(
            by='investment_score', 
            key=lambda x: x.str.extract('(\d+\.?\d*)')[0].astype(float),
            ascending=False
        ).head(5)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Building', 'Type', 'Pred Value', 'Score', 'Rec']),
                cells=dict(values=[
                    top_investments['building_id'],
                    top_investments['building_type'],
                    top_investments['predicted_value'],
                    top_investments['investment_score'],
                    top_investments['recommendation']
                ])
            ),
            row=3, col=1
        )
        
        # 8. Building type distribution
        type_counts = building_df['building_type'].value_counts()
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values),
            row=3, col=2
        )
        
        # 9. Investment heatmap
        fig.add_trace(
            go.Heatmap(
                z=features_df.select_dtypes(include=[np.number]).corr().values,
                x=features_df.select_dtypes(include=[np.number]).columns,
                y=features_df.select_dtypes(include=[np.number]).columns,
                colorscale='RdBu'
            ),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1200, 
            title_text='ThermoValue AI: Building Investment Analysis Dashboard',
            showlegend=True
        )
        
        return fig, investment_df, sensor_df

# ==================== EXECUTION ====================
if __name__ == "__main__":
    # Initialize simulator
    simulator = BuildingValuationSimulator(n_buildings=12, n_days=180)
    
    # Initialize ML model
    model = BuildingValuationModel()
    
    # Create dashboard
    dashboard = InvestmentDashboard(simulator, model)
    fig, investment_df, sensor_data = dashboard.create_dashboard()
    
    print("\n" + "="*80)
    print("THERMOVALUE AI - INVESTMENT ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 ANALYZED {len(investment_df)} BUILDINGS")
    print(f"📈 TEMPERATURE READINGS: {len(sensor_data):,} data points")
    
    # Summary statistics
    recommendations = investment_df['recommendation'].value_counts()
    print(f"\n📋 RECOMMENDATION SUMMARY:")
    for rec, count in recommendations.items():
        print(f"   {rec}: {count} buildings")
    
    avg_efficiency = investment_df['efficiency_score'].str.extract('(\d+\.?\d*)')[0].astype(float).mean()
    print(f"\n⚡ AVERAGE EFFICIENCY SCORE: {avg_efficiency:.1f}/100")
    
    # Show top recommendations
    print(f"\n🏆 TOP 3 INVESTMENT OPPORTUNITIES:")
    top_3 = investment_df.sort_values(
        by='investment_score', 
        key=lambda x: x.str.extract('(\d+\.?\d*)')[0].astype(float),
        ascending=False
    ).head(3)
    
    for _, row in top_3.iterrows():
        print(f"\n   Building {row['building_id']} ({row['building_type']}):")
        print(f"   Value: {row['predicted_value']} (Adj: {row['value_adjustment']})")
        print(f"   Score: {row['investment_score']} - {row['recommendation']}")
        print(f"   Reason: {row['reasoning']}")
    
    print("\n" + "="*80)
    print("Note: This is a simulation for demonstration purposes.")
    print("Real investment decisions require additional due diligence.")
    print("="*80)
    
    # Show dashboard
    fig.show()
    
    # Export results
    investment_df.to_csv('building_investment_analysis.csv', index=False)
    print("\n💾 Results exported to 'building_investment_analysis.csv'")
    
    
    

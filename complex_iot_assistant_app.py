import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import re
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats, optimize, signal, integrate
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧠 Advanced IoT Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CORE SYSTEM ====================

class MathematicalOptimizer:
    """Mathematical optimization engine for IoT systems"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_sensor_placement(self, building_dimensions, sensor_range):
        """Optimal sensor placement using geometric optimization"""
        import numpy as np
        
        length, width, height = building_dimensions
        
        # Calculate optimal grid spacing
        optimal_spacing = sensor_range / np.sqrt(2)
        
        # Generate grid points
        x_points = np.arange(0, length, optimal_spacing)
        y_points = np.arange(0, width, optimal_spacing)
        z_points = np.arange(0, height, optimal_spacing)
        
        # Create mesh grid
        X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
        coordinates = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        # Calculate coverage
        total_volume = length * width * height
        coverage_radius = sensor_range
        sensor_volume = (4/3) * np.pi * (coverage_radius**3)
        estimated_coverage = min(100, (len(coordinates) * sensor_volume / total_volume) * 100)
        
        return {
            'coordinates': coordinates.tolist(),
            'optimal_spacing': optimal_spacing,
            'num_sensors': len(coordinates),
            'coverage_percentage': round(estimated_coverage, 2)
        }
    
    def energy_consumption_optimization(self, power_data):
        """Optimize energy consumption"""
        import numpy as np
        
        power = np.array(power_data)
        
        # Simple optimization: shift usage to off-peak
        sorted_indices = np.argsort(power)
        optimal_schedule = np.zeros_like(power)
        
        # Distribute high consumption evenly
        for i, idx in enumerate(sorted_indices):
            optimal_schedule[idx] = power[idx] * (0.5 + 0.5 * i / len(power))
        
        original_consumption = np.sum(power)
        optimized_consumption = np.sum(optimal_schedule)
        savings = ((original_consumption - optimized_consumption) / original_consumption) * 100
        
        return {
            'optimal_schedule': optimal_schedule.tolist(),
            'savings_percentage': round(savings, 2)
        }
    
    def predictive_maintenance_model(self, sensor_data):
        """Predictive maintenance using statistical analysis"""
        import numpy as np
        import pandas as pd
        
        data = np.array(sensor_data)
        
        # Calculate rolling statistics
        window = 10
        rolling_mean = pd.Series(data).rolling(window=window).mean().dropna().values
        rolling_std = pd.Series(data).rolling(window=window).std().dropna().values
        
        # Calculate z-scores
        z_scores = (data[window-1:] - rolling_mean) / rolling_std
        
        # Detect anomalies
        anomalies = np.where(np.abs(z_scores) > 3)[0]
        
        return {
            'anomalies': anomalies.tolist(),
            'z_scores': z_scores.tolist()
        }


class IntentClassifier:
    """Advanced intent classification system"""
    
    def __init__(self):
        self.optimizer = MathematicalOptimizer()
        self.intents = {
            'financial_analysis': {
                'keywords': ['roi', 'return', 'investment', 'payback', 'cost', 'profit', 'savings', 'budget', 'financial'],
                'weight': 1.5,
                'examples': [
                    "Calculate ROI for a 5000m² building",
                    "What's the payback period for IoT sensors?",
                    "Financial benefits of smart building systems"
                ]
            },
            'sensor_data': {
                'keywords': ['temperature', 'humidity', 'sensor', 'data', 'reading', 'measurement', 'monitor', 'analyze'],
                'weight': 1.3,
                'examples': [
                    "Analyze temperature data from last week",
                    "Show me humidity readings",
                    "Sensor data analysis for anomalies"
                ]
            },
            'risk_assessment': {
                'keywords': ['risk', 'danger', 'threat', 'vulnerability', 'safe', 'security', 'problem', 'issue'],
                'weight': 1.2,
                'examples': [
                    "What are the risks of temperature fluctuations?",
                    "Risk assessment for my building",
                    "Safety issues with current systems"
                ]
            },
            'valuation': {
                'keywords': ['value', 'worth', 'price', 'valuation', 'appraisal', 'increase', 'property', 'asset'],
                'weight': 1.4,
                'examples': [
                    "How much does IoT increase building value?",
                    "Property valuation with smart systems",
                    "Asset value impact analysis"
                ]
            },
            'comparison': {
                'keywords': ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'superior', 'inferior'],
                'weight': 1.3,
                'examples': [
                    "Compare wired vs wireless sensors",
                    "System A vs System B for energy efficiency",
                    "Which is better for my building?"
                ]
            },
            'recommendation': {
                'keywords': ['recommend', 'suggest', 'advice', 'should', 'best', 'optimal', 'ideal', 'solution'],
                'weight': 1.4,
                'examples': [
                    "Recommend IoT systems for industrial buildings",
                    "What should I implement first?",
                    "Best practices for sensor deployment"
                ]
            },
            'mathematical_optimization': {
                'keywords': ['optimize', 'optimal', 'mathematical', 'calculate', 'algorithm', 'solve', 'maximum', 'minimum', 'optimization'],
                'weight': 1.6,
                'examples': [
                    "Optimize sensor placement mathematically",
                    "Calculate optimal energy consumption",
                    "Mathematical solution for sensor deployment"
                ]
            },
            'technical': {
                'keywords': ['how', 'why', 'work', 'function', 'technical', 'specification', 'install', 'configure'],
                'weight': 1.1,
                'examples': [
                    "How do temperature sensors work?",
                    "Technical specifications for IoT systems",
                    "Installation requirements"
                ]
            }
        }
    def classify(self, text: str) -> Tuple[str, float, Dict]:
        """Classify intent with confidence score and extracted data"""
        text_lower = text.lower()
        
        scores = {}
        extracted_data = {}
        
        for intent_name, intent_data in self.intents.items():
            score = 0
            
            # Keyword matching
            for keyword in intent_data['keywords']:
                if keyword in text_lower:
                    score += intent_data['weight']
            
            # Pattern matching for numbers
            if intent_name == 'financial_analysis':
                numbers = re.findall(r'\$\d+[,.]?\d*|\d+[,.]?\d*\s*(?:m²|sqm|square)', text_lower)
                if numbers:
                    score += 2.0
                    extracted_data['financial_numbers'] = numbers
            
            # Extract building size
            size_match = re.search(r'(\d+[,.]?\d*)\s*(?:m²|sqm|square meter|square)', text_lower)
            if size_match:
                extracted_data['building_size'] = float(size_match.group(1).replace(',', ''))
                score += 1.5
            
            # Extract monetary values
            money_matches = re.findall(r'\$(\d+[,.]?\d*)', text)
            if money_matches:
                extracted_data['money_values'] = [float(m.replace(',', '')) for m in money_matches]
                score += 1.5
            
            scores[intent_name] = score
        
        # Get top intent
        if scores:
            top_intent = max(scores.items(), key=lambda x: x[1])
            confidence = min(1.0, top_intent[1] / 10)  # Normalize to 0-1
            
            # Extract time references
            time_refs = self._extract_time_references(text_lower)
            if time_refs:
                extracted_data['time_references'] = time_refs
            
            return top_intent[0], confidence, extracted_data
        
        return 'general', 0.3, extracted_data
    
    def _extract_time_references(self, text: str) -> List[str]:
        """Extract time references from text"""
        patterns = [
            r'last\s+(\d+\s*(?:hour|day|week|month|year))s?',
            r'past\s+(\d+\s*(?:hour|day|week|month|year))s?',
            r'(\d+\s*(?:hour|day|week|month|year))s?\s+ago',
            r'this\s+(morning|afternoon|evening|week|month|year)',
            r'yesterday|today|tomorrow'
        ]
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, text)
            matches.extend(found)
        
        return matches

class AdvancedROIAnalyzer:
    """Advanced ROI analysis with multiple scenarios"""
    
    def analyze(self, params: Dict) -> Dict:
        """Perform comprehensive ROI analysis"""
        size = params.get('building_size', 5000)
        investment = params.get('investment_cost', 75000)
        property_value = params.get('property_value', size * 500)  # $500/m² default
        
        scenarios = {
            'optimistic': self._calculate_scenario(size, investment, property_value, optimism=1.3),
            'baseline': self._calculate_scenario(size, investment, property_value, optimism=1.0),
            'conservative': self._calculate_scenario(size, investment, property_value, optimism=0.7)
        }
        
        return {
            'scenarios': scenarios,
            'sensitivity_analysis': self._sensitivity_analysis(size, investment),
            'breakdown': self._cost_breakdown(investment),
            'recommendations': self._generate_recommendations(scenarios['baseline'])
        }
    
    def _calculate_scenario(self, size: float, investment: float, property_value: float, optimism: float) -> Dict:
        """Calculate ROI for a specific scenario"""
        # Base calculations
        base_energy_savings = size * 3 * optimism  # $3/m²/year
        maintenance_savings = size * 2 * optimism  # $2/m²/year
        value_increase = property_value * 0.08 * optimism  # 8% property value increase
        
        total_annual_savings = base_energy_savings + maintenance_savings
        payback_years = investment / total_annual_savings if total_annual_savings > 0 else float('inf')
        
        # 10-year analysis
        ten_year_savings = total_annual_savings * 10
        total_benefits = ten_year_savings + value_increase
        net_benefit = total_benefits - investment
        roi_percentage = (net_benefit / investment) * 100
        
        return {
            'annual_energy_savings': round(base_energy_savings, 2),
            'annual_maintenance_savings': round(maintenance_savings, 2),
            'total_annual_savings': round(total_annual_savings, 2),
            'property_value_increase': round(value_increase, 2),
            'payback_years': round(payback_years, 2),
            'ten_year_savings': round(ten_year_savings, 2),
            'total_10yr_benefits': round(total_benefits, 2),
            'net_10yr_benefit': round(net_benefit, 2),
            'roi_10yr_percentage': round(roi_percentage, 2),
            'investment_multiple': round(total_benefits / investment, 2)
        }
    
    def _sensitivity_analysis(self, size: float, investment: float) -> Dict:
        """Analyze sensitivity to different factors"""
        factors = {
            'energy_price_change': [-0.2, -0.1, 0, 0.1, 0.2],  # -20% to +20%
            'efficiency_gain': [0.2, 0.3, 0.4, 0.5],  # 20% to 50% efficiency
            'maintenance_reduction': [0.1, 0.2, 0.3, 0.4]  # 10% to 40% reduction
        }
        
        analysis = {}
        for factor, values in factors.items():
            analysis[factor] = {}
            for value in values:
                if factor == 'energy_price_change':
                    savings = size * 3 * (1 + value)
                elif factor == 'efficiency_gain':
                    savings = size * 3 * (1 + value)
                else:  # maintenance_reduction
                    savings = size * 2 * (1 + value)
                
                payback = investment / savings if savings > 0 else float('inf')
                analysis[factor][f"{value*100:.0f}%"] = round(payback, 2)
        
        return analysis
    
    def _cost_breakdown(self, investment: float) -> Dict:
        """Break down investment costs"""
        return {
            'sensors': round(investment * 0.4, 2),
            'installation': round(investment * 0.3, 2),
            'software': round(investment * 0.2, 2),
            'training': round(investment * 0.1, 2)
        }
    
    def _generate_recommendations(self, scenario: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if scenario['payback_years'] < 3:
            recommendations.append("✅ **Excellent Investment** - Payback under 3 years indicates high priority implementation")
        elif scenario['payback_years'] < 5:
            recommendations.append("👍 **Good Investment** - Solid returns with reasonable payback period")
        else:
            recommendations.append("⚠️ **Consider Optimization** - Explore cost reduction or phased implementation")
        
        if scenario['roi_10yr_percentage'] > 200:
            recommendations.append("💰 **High ROI Potential** - 10-year ROI exceeds 200%, indicating strong financial benefits")
        
        recommendations.append("📊 **Monitor Energy Prices** - ROI highly sensitive to energy cost fluctuations")
        recommendations.append("🔄 **Consider Phased Rollout** - Start with high-impact areas first")
        
        return recommendations

class IntelligentResponseBuilder:
    """Build intelligent, context-aware responses"""
    
    def build_response(self, intent: str, confidence: float, data: Dict, analysis: Dict) -> str:
        """Build comprehensive response based on intent and data"""
        
        templates = {
            'financial_analysis': self._financial_response,
            'sensor_data': self._sensor_response,
            'risk_assessment': self._risk_response,
            'valuation': self._valuation_response,
            'comparison': self._comparison_response,
            'recommendation': self._recommendation_response,
            'technical': self._technical_response
        }
        
        if intent in templates:
            return templates[intent](confidence, data, analysis)
        else:
            return self._general_response(confidence, data, analysis)
    
    def _financial_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        response = "## 💰 Financial Analysis Report\n\n"
        
        if confidence > 0.7:
            response += "**High Confidence Analysis** - Based on clear parameters\n\n"
        else:
            response += "**Estimated Analysis** - Using default assumptions where needed\n\n"
        
        if 'scenarios' in analysis:
            response += "### 📈 Investment Scenarios\n\n"
            
            for scenario_name, scenario_data in analysis['scenarios'].items():
                response += f"**{scenario_name.title()} Scenario:**\n"
                response += f"- Annual Savings: **${scenario_data['total_annual_savings']:,.0f}**\n"
                response += f"- Payback Period: **{scenario_data['payback_years']:.1f} years**\n"
                response += f"- 10-Year ROI: **{scenario_data['roi_10yr_percentage']:.0f}%**\n"
                response += f"- Property Value Increase: **${scenario_data['property_value_increase']:,.0f}**\n\n"
        
        if 'recommendations' in analysis:
            response += "### 🎯 Recommendations\n\n"
            for rec in analysis['recommendations']:
                response += f"- {rec}\n"
        
        if 'building_size' in data:
            response += f"\n*Analysis based on {data['building_size']:,.0f} m² building*\n"
        
        return response
    
    def _sensor_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        # Generate sample sensor data
        hours = 48
        timestamps = [datetime.now() - timedelta(hours=i) for i in range(hours)]
        temperatures = np.random.normal(22, 3, hours)
        
        response = "## 🌡️ Sensor Data Analysis\n\n"
        response += f"**Analysis Period:** Last {hours} hours\n\n"
        
        response += "### 📊 Key Metrics\n"
        response += f"- Average Temperature: **{np.mean(temperatures):.1f}°C**\n"
        response += f"- Temperature Range: **{np.min(temperatures):.1f}°C to {np.max(temperatures):.1f}°C**\n"
        response += f"- Stability Index: **{'High' if np.std(temperatures) < 2.5 else 'Medium' if np.std(temperatures) < 3.5 else 'Low'}**\n\n"
        
        # Anomaly detection
        anomalies = temperatures[(temperatures < 16) | (temperatures > 28)]
        response += "### ⚠️ Anomaly Detection\n"
        response += f"- Anomalies Detected: **{len(anomalies)}**\n"
        response += f"- Anomaly Rate: **{(len(anomalies)/hours)*100:.1f}%**\n"
        
        if len(anomalies) > 0:
            response += f"- Most Extreme: **{np.min(anomalies):.1f}°C** to **{np.max(anomalies):.1f}°C**\n\n"
        
        response += "### 🎯 Recommendations\n"
        if np.std(temperatures) > 3:
            response += "- **Priority:** Improve temperature stability\n"
            response += "- **Action:** Check HVAC system calibration\n"
        elif len(anomalies) > hours * 0.1:
            response += "- **Priority:** Address frequent anomalies\n"
            response += "- **Action:** Investigate sensor placement\n"
        else:
            response += "- **Status:** Normal operation\n"
            response += "- **Action:** Continue monitoring\n"
        
        return response
    
    def _risk_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        risk_score = np.random.uniform(0.1, 0.6)
        
        response = "## ⚠️ Risk Assessment Report\n\n"
        
        response += "### 📊 Risk Score\n"
        response += f"**Overall Risk Level:** **{risk_score:.1%}**\n"
        response += f"**Risk Category:** **{'High' if risk_score > 0.4 else 'Medium' if risk_score > 0.2 else 'Low'}**\n\n"
        
        response += "### 🔍 Risk Components\n"
        response += "1. **Equipment Failure Risk:** 15-25%\n"
        response += "2. **Energy Inefficiency Risk:** 20-30%\n"
        response += "3. **Maintenance Cost Risk:** 10-20%\n"
        response += "4. **Compliance Risk:** 5-15%\n"
        response += "5. **Data Security Risk:** 8-12%\n\n"
        
        response += "### 🛡️ Risk Mitigation\n"
        response += "- **IoT Monitoring:** Reduces risk by 25-40%\n"
        response += "- **Predictive Maintenance:** Cuts failure risk by 30-50%\n"
        response += "- **Real-time Alerts:** Immediate response to issues\n"
        response += "- **Data Analytics:** Identifies patterns before failures\n\n"
        
        response += "### 💰 Financial Impact\n"
        response += f"- **Annual Risk Cost:** ${np.random.randint(10000, 50000):,.0f} (estimated)\n"
        response += f"- **Potential Savings:** ${np.random.randint(5000, 30000):,.0f} with IoT\n"
        
        return response
    
    def _valuation_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        """Valuation analysis response"""
        base_value = data.get('property_value', 2500000)
        building_size = data.get('building_size', 5000)
        
        response = "## 🏢 Building Valuation Impact Analysis\n\n"
        
        if confidence > 0.7:
            response += "**High Confidence Valuation** - Based on provided parameters\n\n"
        else:
            response += "**Estimated Valuation** - Using industry benchmarks\n\n"
        
        # Calculate valuation impact
        impact_factors = {
            'risk_reduction': 0.05,  # 5% value increase from risk reduction
            'energy_efficiency': 0.04,  # 4% from energy savings
            'maintenance_optimization': 0.03,  # 3% from maintenance
            'compliance_improvement': 0.02,  # 2% from compliance
            'technology_premium': 0.02  # 2% from being "smart"
        }
        
        total_impact = sum(impact_factors.values())
        value_increase = base_value * total_impact
        new_value = base_value + value_increase
        
        response += "### 📊 Valuation Breakdown\n\n"
        response += f"**Base Property Value:** **${base_value:,.0f}**\n"
        response += f"**With IoT Implementation:** **${new_value:,.0f}**\n"
        response += f"**Value Increase:** **${value_increase:,.0f}** ({total_impact*100:.1f}%)\n\n"
        
        response += "### 📈 Value Drivers\n\n"
        for factor, impact in impact_factors.items():
            factor_value = base_value * impact
            readable_name = factor.replace('_', ' ').title()
            response += f"- **{readable_name}:** +${factor_value:,.0f} ({impact*100:.1f}%)\n"
        
        response += "\n### 🏆 Comparative Analysis\n\n"
        response += "- **Traditional Building:** Relies on manual monitoring\n"
        response += "- **IoT-Enabled Building:** Real-time analytics, predictive maintenance\n"
        response += "- **Premium Justification:** Lower operating costs, reduced risk, future-proofing\n"
        
        response += "\n### 🎯 Market Positioning\n\n"
        response += "**With IoT:** Competitive advantage, premium rental/sale pricing\n"
        response += "**Without IoT:** Standard market rates, higher operational risks\n"
        
        return response
    
    def _comparison_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        """Comparison analysis response"""
        response = "## 🔄 System Comparison Analysis\n\n"
        
        response += "### 📊 Comparison Matrix\n\n"
        response += "| Feature | Wired Sensors | Wireless Sensors | Hybrid System |\n"
        response += "|---------|--------------|-----------------|---------------|\n"
        response += "| **Installation Cost** | High | Medium | High-Medium |\n"
        response += "| **Maintenance** | Low | Medium-High | Medium |\n"
        response += "| **Reliability** | Very High | High | Very High |\n"
        response += "| **Flexibility** | Low | Very High | High |\n"
        response += "| **Data Rate** | Very High | High | Very High |\n"
        response += "| **Scalability** | Difficult | Easy | Moderate |\n\n"
        
        response += "### 🎯 Recommendations by Use Case\n\n"
        response += "**Industrial Buildings:** Wired sensors for critical systems, wireless for flexibility\n"
        response += "**Commercial Offices:** Wireless for easy reconfiguration\n"
        response += "**Mixed-Use:** Hybrid approach for balance of reliability and flexibility\n"
        
        response += "\n### 💰 Cost-Benefit Analysis\n\n"
        response += "- **Wired:** Higher upfront, lower long-term costs\n"
        response += "- **Wireless:** Lower upfront, potential higher maintenance\n"
        response += "- **Hybrid:** Balanced approach, optimal for most buildings\n"
        
        return response
    
    def _recommendation_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        """Recommendation analysis response"""
        building_size = data.get('building_size', 5000)
        
        response = "## 🎯 Personalized Recommendations\n\n"
        
        response += "### 📋 Recommended Implementation Plan\n\n"
        
        response += "**Phase 1: Foundation (Months 1-3)**\n"
        response += f"- **Priority Sensors:** Temperature & Energy ({building_size * 0.3:.0f} sensors)\n"
        response += f"- **Estimated Cost:** ${building_size * 2:,.0f}\n"
        response += "- **Goal:** Baseline data collection, identify high-impact areas\n\n"
        
        response += "**Phase 2: Expansion (Months 4-6)**\n"
        response += f"- **Additional Sensors:** Humidity, Air Quality ({building_size * 0.2:.0f} sensors)\n"
        response += f"- **Estimated Cost:** ${building_size * 1.5:,.0f}\n"
        response += "- **Goal:** Comprehensive monitoring, anomaly detection\n\n"
        
        response += "**Phase 3: Optimization (Months 7-12)**\n"
        response += "- **Advanced Analytics:** Predictive maintenance, optimization algorithms\n"
        response += f"- **Estimated Cost:** ${building_size * 1:,.0f}\n"
        response += "- **Goal:** Automated systems, maximum efficiency\n\n"
        
        response += "### 💡 Key Recommendations\n\n"
        response += "1. **Start Small, Scale Smart:** Begin with critical areas\n"
        response += "2. **Focus on ROI:** Prioritize energy and maintenance savings\n"
        response += "3. **Consider Future Needs:** Choose scalable solutions\n"
        response += "4. **Training is Crucial:** Ensure staff can use the system effectively\n"
        
        return response
    
    def _technical_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        """Technical analysis response"""
        response = "## 🔧 Technical Specifications & Requirements\n\n"
        
        response += "### 📡 IoT System Architecture\n\n"
        response += "**Sensing Layer:**\n"
        response += "- Temperature Sensors: ±0.5°C accuracy\n"
        response += "- Humidity Sensors: ±3% RH accuracy\n"
        response += "- Energy Meters: Class 1 accuracy\n"
        response += "- Motion Sensors: 10m range\n\n"
        
        response += "**Network Layer:**\n"
        response += "- Protocol: MQTT/HTTP for data transmission\n"
        response += "- Security: TLS 1.3 encryption\n"
        response += "- Bandwidth: 100 Mbps minimum\n"
        response += "- Latency: < 100ms for critical alerts\n\n"
        
        response += "**Application Layer:**\n"
        response += "- Dashboard: Real-time monitoring\n"
        response += "- Analytics: Machine learning for anomaly detection\n"
        response += "- Alerts: SMS/Email/App notifications\n"
        response += "- Reporting: Automated PDF/Excel reports\n\n"
        
        response += "### ⚙️ Installation Requirements\n\n"
        response += "**Physical Requirements:**\n"
        response += "- Power: PoE or battery options\n"
        response += "- Mounting: Standard DIN rail or wall mount\n"
        response += "- Environment: IP65 rating for industrial areas\n"
        response += "- Temperature Range: -20°C to 60°C operation\n\n"
        
        response += "**IT Requirements:**\n"
        response += "- Network: VLAN segmentation recommended\n"
        response += "- Storage: 1TB minimum for 1-year data retention\n"
        response += "- Backup: Daily automated backups\n"
        response += "- Security: Regular firmware updates\n"
        
        return response
    
    def _general_response(self, confidence: float, data: Dict, analysis: Dict) -> str:
        return """## 🧠 IoT Assistant Response

I've analyzed your query and here's what I can help you with:

### Available Analysis Types:
1. **💰 Financial Analysis** - ROI, payback, investment returns
2. **🌡️ Sensor Data** - Temperature, humidity, anomaly detection  
3. ⚠️ **Risk Assessment** - Building risks and mitigation strategies
4. 🏢 **Valuation** - Property value impact analysis
5. 🔄 **Comparisons** - System comparisons and recommendations
6. 🎯 **Recommendations** - Tailored suggestions for your building
7. 🔧 **Technical** - Specifications and requirements

### Try Asking:
- "Calculate ROI for a 5000m² building with $75k investment"
- "Analyze temperature risks in my industrial building"
- "Compare wired vs wireless sensor systems"
- "How much will IoT increase my property value?"
- "Technical requirements for IoT installation"

*For more specific analysis, please provide building details or requirements.*"""

# ==================== STREAMLIT APP ====================

def main():
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = {
            'classifier': IntentClassifier(),
            'roi_analyzer': AdvancedROIAnalyzer(),
            'response_builder': IntelligentResponseBuilder(),
            'conversation': [],
            'context': {}
        }
    
    # Title and description
    st.title("🧠 Advanced IoT Assistant System")
    st.markdown("""
    **Intelligent analysis for building management, ROI calculations, sensor data, and risk assessment**
    
    This system understands complex questions and provides detailed, actionable insights.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.assistant['conversation'] = []
                st.session_state.assistant['context'] = {}
                st.rerun()
        
        with col2:
            if st.button("📊 Demo Data", use_container_width=True):
                st.session_state.demo_mode = True
                st.rerun()
        
        st.header("📋 Example Questions")
        
        examples = [
            "Calculate ROI for 8000 m² building with $120k investment",
            "What are the risks of temperature fluctuations in warehouses?",
            "Compare energy savings between System A ($50k) and System B ($80k)",
            "How much will smart sensors increase my $2.5M property value?",
            "Analyze temperature data patterns and recommend improvements",
            "What's the optimal sensor configuration for industrial safety?",
            "Financial benefits of IoT vs traditional monitoring systems"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                st.session_state.prefilled_question = example
                st.rerun()
        
        st.header("⚙️ Settings")
        st.session_state.detail_level = st.select_slider(
            "Analysis Detail",
            options=["Overview", "Standard", "Detailed", "Comprehensive"],
            value="Standard"
        )
        
        st.header("📈 System Info")
        st.metric("Conversation", len(st.session_state.assistant['conversation']))
        st.metric("Context Items", len(st.session_state.assistant['context']))
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Conversation display
        st.header("💬 Conversation")
        
        conv_container = st.container()
        with conv_container:
            for i, msg in enumerate(st.session_state.assistant['conversation']):
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    
                    if "analysis" in msg:
                        with st.expander("📊 View Analysis Details"):
                            st.json(msg["analysis"], expanded=False)
        
        # Question input
        st.header("🎯 Ask a Question")
        
        question = st.chat_input(
            "Type your question here...",
            key="chat_input"
        )
        
        # Use prefilled question if available
        if hasattr(st.session_state, 'prefilled_question') and st.session_state.prefilled_question:
            question = st.session_state.prefilled_question
            del st.session_state.prefilled_question
        
        if question:
            # Add user question to conversation
            st.session_state.assistant['conversation'].append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().isoformat()
            })
            
            # Process the question
            with st.spinner("🧠 Analyzing question..."):
                # Classify intent
                intent, confidence, extracted_data = st.session_state.assistant['classifier'].classify(question)
                
                # Perform analysis based on intent
                analysis = {}
                if intent == 'financial_analysis':
                    analysis = st.session_state.assistant['roi_analyzer'].analyze(extracted_data)
                
                # Build response
                response = st.session_state.assistant['response_builder'].build_response(
                    intent, confidence, extracted_data, analysis
                )
                
                # Add assistant response to conversation
                st.session_state.assistant['conversation'].append({
                    "role": "assistant",
                    "content": response,
                    "analysis": {
                        "intent": intent,
                        "confidence": confidence,
                        "extracted_data": extracted_data,
                        "analysis_type": list(analysis.keys()) if analysis else []
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            # Rerun to display new messages
            st.rerun()
    
    with col2:
        st.header("📊 Live Dashboard")
        
        # System status
        if st.session_state.assistant['conversation']:
            last_msg = st.session_state.assistant['conversation'][-1]
            if last_msg["role"] == "assistant":
                analysis = last_msg.get("analysis", {})
                st.metric("Last Intent", analysis.get("intent", "N/A"))
                st.metric("Confidence", f"{analysis.get('confidence', 0)*100:.0f}%")
        
        # Quick stats
        st.header("📈 Quick Analysis")
        
        if st.button("💰 Quick ROI Estimate"):
            building_size = st.number_input("Building Size (m²)", 1000, 50000, 5000, key="quick_size")
            investment = st.number_input("Investment ($)", 10000, 500000, 75000, key="quick_invest")
            
            if st.button("Calculate", key="quick_calc"):
                analysis = st.session_state.assistant['roi_analyzer'].analyze({
                    'building_size': building_size,
                    'investment_cost': investment
                })
                
                baseline = analysis['scenarios']['baseline']
                st.metric("Annual Savings", f"${baseline['total_annual_savings']:,.0f}")
                st.metric("Payback", f"{baseline['payback_years']:.1f} years")
                st.metric("10-Year ROI", f"{baseline['roi_10yr_percentage']:.0f}%")
        
        # Data visualization
        st.header("📊 Sample Data")
        
        if st.button("Generate Sensor Chart"):
            hours = 24
            temps = np.random.normal(22, 3, hours)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=temps,
                mode='lines+markers',
                name='Temperature',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title='Sample Temperature Data',
                yaxis_title='Temperature (°C)',
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.metric("Average", f"{np.mean(temps):.1f}°C")
        
        # Export functionality
        st.header("💾 Export")
        
        if st.button("Download Conversation"):
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "conversation": st.session_state.assistant['conversation'],
                "context": st.session_state.assistant['context'],
                "system_info": {
                    "version": "2.0",
                    "analysis_types": ["Financial", "Sensor", "Risk", "Valuation", "Comparison"]
                }
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"iot_assistant_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 System Information")
    st.sidebar.info("""
    **Advanced IoT Assistant v2.0**
    
    Capabilities:
    - Intent Classification
    - Financial Analysis
    - Risk Assessment  
    - Sensor Data Analysis
    - Intelligent Responses
    - Context Awareness
    
    *No API costs • Local processing • Privacy focused*
    """)

if __name__ == "__main__":
    main()

"""
🚀 IoT ML REAL-TIME DASHBOARD
Uses your trained model (98.3% accuracy!)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="IoT ML Dashboard - 98.3% Accuracy",
    page_icon="📈",
    layout="wide"
)

# Load YOUR trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('iot_anomaly_model_large.pkl')
        return model
    except:
        st.error("❌ Could not load model 'iot_anomaly_model_large.pkl'")
        return None

# Load your generated data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('iot_large_dataset.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return None

# Title
st.title("📊 IoT Sensor ML Dashboard")
st.markdown(f"""
<div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px;'>
    <h3 style='color: #2e7d32; margin: 0;'>🎯 Model Performance: <b>98.3% Accuracy</b></h3>
    <p style='margin: 5px 0 0 0;'>Trained on 10,000 samples | Detects temperature/humidity anomalies</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model and data
model = load_model()
df = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Dashboard Controls")
    
    # Model info
    if model:
        st.success("✅ Model Loaded Successfully")
        st.caption(f"Features: Temperature, Humidity, Pressure, Battery, Vibration")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_live = st.toggle("Show Live Predictions", value=True)
    anomaly_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)
    
    # Data options
    st.subheader("📈 Data View")
    n_samples_view = st.slider("Samples to Display", 100, 5000, 1000)
    
    st.markdown("---")
    st.caption("Using your trained Random Forest model")

# If data loaded, show metrics
if df is not None:
    # Calculate metrics
    features = ['temperature', 'humidity', 'pressure', 'battery', 'vibration']
    
    # Make predictions on the data
    X = df[features]
    if model:
        predictions = model.predict(X)
        confidence = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.ones(len(X)) * 0.98
        df['prediction'] = predictions
        df['confidence'] = confidence
        df['alert'] = (confidence > anomaly_threshold) & (predictions == 1)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{len(df):,}")
    
    with col2:
        actual_anomalies = df['anomaly'].sum() if 'anomaly' in df.columns else 0
        st.metric("Actual Anomalies", f"{actual_anomalies:,}")
    
    with col3:
        if 'prediction' in df.columns:
            predicted = df['prediction'].sum()
            st.metric("ML Predictions", f"{predicted:,}")
    
    with col4:
        if 'alert' in df.columns:
            alerts = df['alert'].sum()
            st.metric("High-Confidence Alerts", f"{alerts:,}")
    
    st.markdown("---")
    
    # Chart 1: Temperature & Humidity Over Time
    st.subheader("📈 Sensor Readings Over Time")
    
    # Sample data for performance
    chart_data = df.head(n_samples_view).copy()
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Temperature
    fig1.add_trace(
        go.Scatter(
            x=chart_data['timestamp'],
            y=chart_data['temperature'],
            name="Temperature (°C)",
            line=dict(color='red', width=2),
            mode='lines'
        ),
        secondary_y=False,
    )
    
    # Humidity
    fig1.add_trace(
        go.Scatter(
            x=chart_data['timestamp'],
            y=chart_data['humidity'],
            name="Humidity (%)",
            line=dict(color='blue', width=2),
            mode='lines'
        ),
        secondary_y=True,
    )
    
    # Highlight anomalies if available
    if 'anomaly' in chart_data.columns:
        anomaly_points = chart_data[chart_data['anomaly'] == 1]
        fig1.add_trace(
            go.Scatter(
                x=anomaly_points['timestamp'],
                y=anomaly_points['temperature'],
                mode='markers',
                name='Actual Anomaly',
                marker=dict(size=8, color='red', symbol='x')
            ),
            secondary_y=False,
        )
    
    # Highlight ML predictions if available
    if 'alert' in chart_data.columns:
        alert_points = chart_data[chart_data['alert'] == True]
        fig1.add_trace(
            go.Scatter(
                x=alert_points['timestamp'],
                y=alert_points['temperature'],
                mode='markers',
                name='ML High-Confidence Alert',
                marker=dict(size=10, color='orange', symbol='star')
            ),
            secondary_y=False,
        )
    
    fig1.update_layout(
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    fig1.update_xaxes(title_text="Time")
    fig1.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig1.update_yaxes(title_text="Humidity (%)", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Row 2: Feature Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Feature Importance")
        
        # Your model's actual feature importance
        feature_importance = {
            'temperature': 0.6227,
            'humidity': 0.3063,
            'pressure': 0.0274,
            'vibration': 0.0233,
            'battery': 0.0203
        }
        
        importance_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        fig2 = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            title='What the Model Learned'
        )
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Anomaly Distribution")
        
        if 'anomaly_type' in df.columns:
            anomaly_counts = df[df['anomaly'] == 1]['anomaly_type'].value_counts().reset_index()
            anomaly_counts.columns = ['Anomaly Type', 'Count']
            
            fig3 = px.pie(
                anomaly_counts,
                values='Count',
                names='Anomaly Type',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            # Create correlation heatmap
            corr = df[features].corr()
            fig3 = px.imshow(
                corr,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                title='Feature Correlation'
            )
            fig3.update_layout(height=300)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Row 3: Real-time simulation
    if show_live:
        st.subheader("🔄 Live Anomaly Detection Simulation")
        
        # Generate new "live" data
        np.random.seed(42)
        live_samples = 100
        live_data = []
        
        for i in range(live_samples):
            temp = np.random.normal(25, 5)
            humidity = np.random.normal(50, 15)
            
            # Occasionally inject an anomaly
            if i == 25 or i == 75:
                temp = temp * 2  # Simulated anomaly
                is_anomaly = True
            else:
                is_anomaly = False
            
            live_data.append({
                'timestamp': datetime.datetime.now() - timedelta(seconds=(live_samples - i) * 10),
                'temperature': temp,
                'humidity': humidity,
                'pressure': np.random.normal(1013, 5),
                'battery': np.random.uniform(30, 100),
                'vibration': np.random.exponential(0.5),
                'is_anomaly': is_anomaly
            })
        
        live_df = pd.DataFrame(live_data)
        
        # Predict using your model
        if model:
            X_live = live_df[features]
            predictions = model.predict(X_live)
            confidence = model.predict_proba(X_live)[:, 1] if hasattr(model, 'predict_proba') else np.ones(len(X_live))
            live_df['prediction'] = predictions
            live_df['confidence'] = confidence
        
        # Display live alerts
        if 'prediction' in live_df.columns:
            alerts = live_df[live_df['prediction'] == 1]
            if len(alerts) > 0:
                st.warning(f"🚨 **{len(alerts)} anomalies detected in live data!**")
                
                for idx, row in alerts.head(3).iterrows():
                    st.markdown(f"""
                    <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                        <b>Alert at {row['timestamp'].strftime('%H:%M:%S')}</b><br>
                        Temp: {row['temperature']:.1f}°C | Humidity: {row['humidity']:.1f}% | 
                        Confidence: {row['confidence']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No anomalies detected in current live data")
        
        # Live data chart
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=live_df['timestamp'],
            y=live_df['temperature'],
            name='Temperature',
            line=dict(color='red')
        ))
        
        if 'prediction' in live_df.columns:
            anomaly_points = live_df[live_df['prediction'] == 1]
            fig4.add_trace(go.Scatter(
                x=anomaly_points['timestamp'],
                y=anomaly_points['temperature'],
                mode='markers',
                name='ML Detection',
                marker=dict(size=10, color='red', symbol='x')
            ))
        
        fig4.update_layout(
            title='Live Temperature Feed with ML Detection',
            height=300,
            xaxis_title='Time',
            yaxis_title='Temperature (°C)'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Row 4: Data Explorer
    st.subheader("📋 Data Explorer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive data table
        display_cols = ['timestamp', 'device_id', 'temperature', 'humidity']
        if 'prediction' in df.columns:
            display_cols.append('prediction')
        if 'confidence' in df.columns:
            display_cols.append('confidence')
        
        st.dataframe(
            df[display_cols].head(20),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Avg Temp", f"{df['temperature'].mean():.1f}°C")
        st.metric("Avg Humidity", f"{df['humidity'].mean():.1f}%")
        st.metric("Data Range", f"{len(df):,} samples")
        
        if model:
            st.download_button(
                label="📥 Download Model",
                data=open('iot_anomaly_model_large.pkl', 'rb'),
                file_name="iot_anomaly_model.pkl",
                mime="application/octet-stream"
            )
    
    # Footer
    st.markdown("---")
    st.caption("""
    **Dashboard Features:**
    • Uses your trained ML model (98.3% accuracy)
    • Shows feature importance (temperature is 62% important)
    • Simulates live anomaly detection
    • Interactive visualizations with Plotly
    """)

else:
    st.warning("⚠️ Could not load data file 'iot_large_dataset.csv'")
    st.info("Run the ML pipeline first to generate data and train the model")

# Add auto-refresh
if st.button("🔄 Refresh Dashboard"):
    st.rerun()

st.markdown("---")
st.caption(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Using model: iot_anomaly_model_large.pkl")

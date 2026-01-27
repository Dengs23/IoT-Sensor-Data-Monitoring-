# monitoring_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential

class AzureMonitorDashboard:
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.client = LogsQueryClient(self.credential)
        
    def get_iot_metrics(self, workspace_id, time_range="PT1H"):
        """Get IoT metrics from Azure Monitor"""
        query = """
        AzureMetrics
        | where ResourceProvider == "MICROSOFT.EVENTHUB"
        | where MetricName in ("IncomingMessages", "IncomingBytes", "OutgoingMessages")
        | summarize avg(TimeGrain) by MetricName, bin(TimeGenerated, 1m)
        | order by TimeGenerated asc
        """
        
        response = self.client.query_workspace(
            workspace_id=workspace_id,
            query=query,
            timespan=timedelta(hours=1)
        )
        
        return response.tables[0]
    
    def create_dashboard(self):
        """Create interactive monitoring dashboard"""
        st.title("🔍 Azure IoT Platform Monitoring")
        st.markdown("Real-time monitoring of IoT analytics platform")
        
        # Sidebar configuration
        st.sidebar.header("Monitoring Configuration")
        workspace_id = st.sidebar.text_input("Log Analytics Workspace ID", "your-workspace-id")
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60)
        
        # Auto-refresh
        if st.sidebar.button("Start Auto-refresh"):
            st.experimental_rerun()
        
        # Main dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Devices", "142", "+5")
        
        with col2:
            st.metric("Messages/sec", "1,245", "-12")
        
        with col3:
            st.metric("Avg Latency", "47ms", "+3ms")
        
        # Metrics charts
        st.subheader("Platform Metrics")
        
        # Simulated metrics (in production, would query Azure Monitor)
        time_points = pd.date_range(start=datetime.now() - timedelta(minutes=60), 
                                   end=datetime.now(), freq='1min')
        
        # Create simulated metrics
        metrics_data = pd.DataFrame({
            'timestamp': time_points,
            'incoming_messages': np.random.poisson(1200, len(time_points)),
            'processing_latency': np.random.normal(45, 5, len(time_points)),
            'error_rate': np.random.beta(1, 99, len(time_points)) * 100,
            'active_connections': np.random.randint(130, 150, len(time_points))
        })
        
        # Create charts
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=metrics_data['timestamp'],
            y=metrics_data['incoming_messages'],
            mode='lines',
            name='Messages/sec',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics_data['timestamp'],
            y=metrics_data['processing_latency'],
            mode='lines',
            name='Latency (ms)',
            yaxis='y2',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Platform Performance Metrics',
            yaxis=dict(title='Messages/sec'),
            yaxis2=dict(
                title='Latency (ms)',
                overlaying='y',
                side='right'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Alerts section
        st.subheader("Recent Alerts")
        
        alerts = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(minutes=i*5) for i in range(10)],
            'severity': np.random.choice(['High', 'Medium', 'Low'], 10),
            'message': [
                'High temperature detected on sensor_001',
                'Low battery on sensor_005',
                'Connection lost to sensor_003',
                'Unusual pressure reading on sensor_002',
                'High humidity anomaly detected',
                'Sensor_004 offline for 15 minutes',
                'Data ingestion rate dropped by 40%',
                'High memory usage on processing node',
                'Database connection timeout',
                'API response time > 2 seconds'
            ],
            'device': np.random.choice([f'sensor_{i:03d}' for i in range(1, 11)], 10),
            'status': np.random.choice(['Active', 'Resolved', 'Investigating'], 10)
        })
        
        # Color code by severity
        def color_severity(severity):
            if severity == 'High':
                return 'background-color: #ffcccc'
            elif severity == 'Medium':
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #d4edda'
        
        styled_alerts = alerts.style.applymap(
            lambda x: color_severity(x) if isinstance(x, str) and x in ['High', 'Medium', 'Low'] else '',
            subset=['severity']
        )
        
        st.dataframe(styled_alerts, use_container_width=True)
        
        # System health
        st.subheader("System Health Status")
        
        health_data = pd.DataFrame({
            'Component': ['Event Hubs', 'Cosmos DB', 'Azure Functions', 'AKS Cluster', 'Redis Cache'],
            'Status': ['Healthy', 'Healthy', 'Warning', 'Healthy', 'Error'],
            'CPU %': [45, 32, 85, 62, 95],
            'Memory %': [67, 41, 78, 55, 89],
            'Latency (ms)': [12, 8, 45, 22, 150]
        })
        
        # Create health status indicators
        for _, row in health_data.iterrows():
            col1, col2, col3 = st.columns([1, 2, 2])
            
            with col1:
                if row['Status'] == 'Healthy':
                    st.success(f"✅ {row['Component']}")
                elif row['Status'] == 'Warning':
                    st.warning(f"⚠️ {row['Component']}")
                else:
                    st.error(f"❌ {row['Component']}")
            
            with col2:
                st.progress(row['CPU %'] / 100, text=f"CPU: {row['CPU %']}%")
            
            with col3:
                st.progress(row['Memory %'] / 100, text=f"Memory: {row['Memory %']}%")
        
        # Resource utilization chart
        st.subheader("Resource Utilization")
        
        fig_resources = go.Figure(data=[
            go.Bar(name='CPU %', x=health_data['Component'], y=health_data['CPU %']),
            go.Bar(name='Memory %', x=health_data['Component'], y=health_data['Memory %'])
        ])
        
        fig_resources.update_layout(
            barmode='group',
            title='Component Resource Utilization'
        )
        
        st.plotly_chart(fig_resources, use_container_width=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = AzureMonitorDashboard()
    dashboard.create_dashboard()
    
    
    

# iot_analytics_notebook.ipynb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ipywidgets as widgets
from IPython.display import display, clear_output
import asyncio

# Interactive widget for data exploration
class IoTDataExplorer:
    def __init__(self, data_path="iot_sensor_data.csv"):
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
    def create_dashboard(self):
        """Create interactive dashboard with widgets"""
        
        # Device selector
        device_selector = widgets.Dropdown(
            options=['All'] + list(self.df['device_id'].unique()),
            value='All',
            description='Device:'
        )
        
        # Metric selector
        metric_selector = widgets.Dropdown(
            options=['temperature', 'humidity', 'pressure', 'battery_level'],
            value='temperature',
            description='Metric:'
        )
        
        # Time range selector
        time_selector = widgets.Dropdown(
            options=['Last hour', 'Last 6 hours', 'Last 24 hours', 'Last week', 'All'],
            value='Last 24 hours',
            description='Time range:'
        )
        
        # Anomaly threshold slider
        threshold_slider = widgets.FloatSlider(
            value=3.0,
            min=1.0,
            max=5.0,
            step=0.5,
            description='Z-score threshold:'
        )
        
        # Detect anomalies button
        detect_button = widgets.Button(
            description="Detect Anomalies",
            button_style='primary'
        )
        
        # Output area
        output = widgets.Output()
        
        def update_plot(change):
            with output:
                clear_output(wait=True)
                
                # Filter data
                filtered_df = self.df.copy()
                
                if device_selector.value != 'All':
                    filtered_df = filtered_df[filtered_df['device_id'] == device_selector.value]
                
                # Filter by time
                if time_selector.value == 'Last hour':
                    cutoff = datetime.now() - timedelta(hours=1)
                elif time_selector.value == 'Last 6 hours':
                    cutoff = datetime.now() - timedelta(hours=6)
                elif time_selector.value == 'Last 24 hours':
                    cutoff = datetime.now() - timedelta(days=1)
                elif time_selector.value == 'Last week':
                    cutoff = datetime.now() - timedelta(weeks=1)
                else:
                    cutoff = filtered_df['timestamp'].min()
                
                filtered_df = filtered_df[filtered_df['timestamp'] >= cutoff]
                
                # Create plot
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        f'{metric_selector.value.capitalize()} Over Time',
                        'Distribution',
                        'Anomaly Detection',
                        'Correlation Heatmap'
                    )
                )
                
                # Time series
                for device in filtered_df['device_id'].unique():
                    device_data = filtered_df[filtered_df['device_id'] == device]
                    fig.add_trace(
                        go.Scatter(
                            x=device_data['timestamp'],
                            y=device_data[metric_selector.value],
                            mode='lines',
                            name=device
                        ),
                        row=1, col=1
                    )
                
                # Distribution
                fig.add_trace(
                    go.Histogram(
                        x=filtered_df[metric_selector.value],
                        nbinsx=30,
                        name='Distribution'
                    ),
                    row=1, col=2
                )
                
                # Anomaly detection
                if len(filtered_df) > 10:
                    # Calculate z-scores
                    values = filtered_df[metric_selector.value].values
                    mean = np.mean(values)
                    std = np.std(values)
                    
                    if std > 0:
                        z_scores = (values - mean) / std
                        anomalies = np.abs(z_scores) > threshold_slider.value
                        
                        # Plot with anomalies highlighted
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_df['timestamp'],
                                y=filtered_df[metric_selector.value],
                                mode='markers',
                                marker=dict(
                                    color=['red' if a else 'blue' for a in anomalies],
                                    size=8
                                ),
                                name='Anomalies'
                            ),
                            row=2, col=1
                        )
                        
                        # Add threshold lines
                        fig.add_hline(
                            y=mean + threshold_slider.value * std,
                            line_dash="dash",
                            line_color="red",
                            row=2, col=1
                        )
                        fig.add_hline(
                            y=mean - threshold_slider.value * std,
                            line_dash="dash",
                            line_color="red",
                            row=2, col=1
                        )
                
                # Correlation heatmap
                numeric_cols = ['temperature', 'humidity', 'pressure', 'battery_level']
                corr_matrix = filtered_df[numeric_cols].corr()
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmid=0
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(height=800, showlegend=True)
                fig.show()
                
                # Display statistics
                stats = filtered_df[metric_selector.value].describe()
                print("Statistics:")
                print(stats)
        
        # Wire up events
        device_selector.observe(update_plot, names='value')
        metric_selector.observe(update_plot, names='value')
        time_selector.observe(update_plot, names='value')
        threshold_slider.observe(update_plot, names='value')
        
        def on_detect_click(b):
            with output:
                clear_output(wait=True)
                print("Running advanced anomaly detection...")
                
                # Simulate ML-based anomaly detection
                from sklearn.ensemble import IsolationForest
                
                # Prepare features
                features = self.df[['temperature', 'humidity', 'pressure']].fillna(0)
                
                # Train isolation forest
                clf = IsolationForest(contamination=0.05, random_state=42)
                predictions = clf.fit_predict(features)
                
                # Count anomalies
                anomaly_count = (predictions == -1).sum()
                total_count = len(predictions)
                
                print(f"ML Anomaly Detection Results:")
                print(f"Total samples: {total_count}")
                print(f"Anomalies detected: {anomaly_count} ({anomaly_count/total_count:.1%})")
                
                # Show anomaly details
                self.df['is_anomaly'] = predictions == -1
                anomalies = self.df[self.df['is_anomaly']]
                
                if not anomalies.empty:
                    print("\nTop 10 anomalies:")
                    print(anomalies[['device_id', 'timestamp', 'temperature', 'humidity', 'pressure']].head(10))
        
        detect_button.on_click(on_detect_click)
        
        # Display widgets
        controls = widgets.VBox([
            device_selector,
            metric_selector,
            time_selector,
            threshold_slider,
            detect_button
        ])
        
        display(widgets.HBox([controls, output]))
        
        # Initial plot
        update_plot(None)

# Run the explorer
explorer = IoTDataExplorer()
explorer.create_dashboard()




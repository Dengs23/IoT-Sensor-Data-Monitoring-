import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Initialize Dash app
app = dash.Dash(__name__, title="IoT Dashboard Pro", suppress_callback_exceptions=True)
server = app.server

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>🚀 IoT Dashboard Pro</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .main-container {
                max-width: 1400px;
                margin: 20px auto;
                padding: 20px;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            .header {
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
                color: white;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .tab-container {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                margin: 10px;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                color: #1E3A8A;
            }
            .metric-label {
                font-size: 1rem;
                color: #666;
            }
            .control-panel {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("🚀 IoT Dashboard Pro", className="header"),
        html.P("Advanced IoT Analytics & Visualization Platform", 
               style={'textAlign': 'center', 'color': 'white', 'marginBottom': '30px'})
    ]),
    
    html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            dcc.Tab(label='📊 3D Network', value='tab-1'),
            dcc.Tab(label='⚡ Energy Analytics', value='tab-2'),
            dcc.Tab(label='💰 ROI Calculator', value='tab-3'),
            dcc.Tab(label='📈 Real-time Monitor', value='tab-4'),
        ]),
        html.Div(id='tabs-content')
    ], className="main-container"),
    
    # Hidden div for storing data
    dcc.Store(id='sensor-data-store'),
    dcc.Store(id='roi-results-store'),
    
    # Interval for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=2000,  # 2 seconds
        n_intervals=0
    )
])

# ==================== 3D VISUALIZATION ====================
def create_3d_network(building_area=1000, num_sensors=20):
    """Create 3D sensor network visualization"""
    length = building_area ** 0.5 * 1.2
    width = building_area ** 0.5 * 0.8
    height = 4
    
    # Generate sensor coordinates
    spacing = 5
    x_points = np.arange(2, length, spacing)
    y_points = np.arange(2, width, spacing)
    z_points = [1.5, 3.0]
    
    coordinates = []
    for x in x_points:
        for y in y_points:
            for z in z_points:
                if len(coordinates) < num_sensors:
                    coordinates.append([x, y, z])
    
    # Create 3D figure
    fig = go.Figure()
    
    # Building wireframe
    x_box = [0, length, length, 0, 0, length, length, 0]
    y_box = [0, 0, width, width, 0, 0, width, width]
    z_box = [0, 0, 0, 0, height, height, height, height]
    
    fig.add_trace(go.Mesh3d(
        x=x_box, y=y_box, z=z_box,
        color='lightblue',
        opacity=0.1,
        name='Building'
    ))
    
    if coordinates:
        coords = np.array(coordinates)
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=coords[:, 2],
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Sensors'
        ))
    
    fig.update_layout(
        title=f"3D Sensor Network: {len(coordinates)} sensors",
        scene=dict(
            xaxis_title='Length (m)',
            yaxis_title='Width (m)',
            zaxis_title='Height (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

# ==================== ROI CALCULATOR ====================
def calculate_roi(investment, building_area, energy_cost, years):
    """Calculate ROI metrics"""
    energy_savings = building_area * 2.5 * energy_cost * 365 * 0.25
    total_savings = energy_savings * years
    net_profit = total_savings - investment
    roi_percentage = (net_profit / investment) * 100 if investment > 0 else 0
    payback_years = investment / energy_savings if energy_savings > 0 else float('inf')
    
    return {
        'roi': round(roi_percentage, 2),
        'payback': round(payback_years, 2),
        'annual_savings': round(energy_savings, 2),
        'net_profit': round(net_profit, 2)
    }

# ==================== CALLBACKS ====================
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H2("📊 3D Sensor Network Visualization", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.Label("Building Area (m²)"),
                    dcc.Slider(
                        id='building-area-slider',
                        min=100,
                        max=5000,
                        value=1000,
                        step=100,
                        marks={100: '100', 2500: '2500', 5000: '5000'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Number of Sensors"),
                    dcc.Slider(
                        id='sensor-count-slider',
                        min=5,
                        max=100,
                        value=20,
                        step=5,
                        marks={5: '5', 50: '50', 100: '100'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Button("🔄 Update Visualization", id='update-3d-btn', n_clicks=0,
                               style={'width': '100%', 'padding': '15px', 'fontSize': '16px'})
                ], className="control-panel"),
            ]),
            
            html.Div([
                dcc.Graph(id='3d-network-graph', figure=create_3d_network())
            ], style={'marginTop': '20px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(id='sensor-count', className="metric-value"),
                        html.Div("Sensors Deployed", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='coverage-percentage', className="metric-value"),
                        html.Div("Estimated Coverage", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='optimal-spacing', className="metric-value"),
                        html.Div("Optimal Spacing", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='building-size', className="metric-value"),
                        html.Div("Building Size", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
            ], className="row", style={'display': 'flex', 'justifyContent': 'space-between'})
        ], className="tab-container")
    
    elif tab == 'tab-2':
        return html.Div([
            html.H2("⚡ Energy Analytics", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Peak Load Adjustment (%)"),
                    dcc.Slider(
                        id='peak-load-slider',
                        min=-50,
                        max=50,
                        value=0,
                        step=5,
                        marks={-50: '-50%', 0: '0%', 50: '50%'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("System Efficiency (%)"),
                    dcc.Slider(
                        id='efficiency-slider',
                        min=60,
                        max=95,
                        value=80,
                        step=5,
                        marks={60: '60%', 80: '80%', 95: '95%'}
                    ),
                ], className="control-panel"),
            ]),
            
            html.Div([
                dcc.Graph(id='energy-heatmap')
            ]),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(id='daily-consumption', className="metric-value"),
                        html.Div("Daily Consumption (kWh)", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='daily-cost', className="metric-value"),
                        html.Div("Daily Cost ($)", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='monthly-savings', className="metric-value"),
                        html.Div("Monthly Savings ($)", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='annual-savings-energy', className="metric-value"),
                        html.Div("Annual Savings ($)", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
            ], className="row", style={'display': 'flex', 'justifyContent': 'space-between'})
        ], className="tab-container")
    
    elif tab == 'tab-3':
        return html.Div([
            html.H2("💰 ROI Calculator", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Investment Amount ($)"),
                    dcc.Input(
                        id='investment-input',
                        type='number',
                        value=50000,
                        min=10000,
                        max=1000000,
                        step=5000,
                        style={'width': '100%', 'padding': '10px'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Building Area (m²)"),
                    dcc.Input(
                        id='roi-area-input',
                        type='number',
                        value=2000,
                        min=100,
                        max=50000,
                        step=100,
                        style={'width': '100%', 'padding': '10px'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Energy Cost ($/kWh)"),
                    dcc.Slider(
                        id='energy-cost-slider',
                        min=0.05,
                        max=0.50,
                        value=0.15,
                        step=0.01,
                        marks={0.05: '$0.05', 0.25: '$0.25', 0.50: '$0.50'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Analysis Period (Years)"),
                    dcc.Slider(
                        id='years-slider',
                        min=1,
                        max=20,
                        value=10,
                        step=1,
                        marks={1: '1', 10: '10', 20: '20'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Button("📊 Calculate ROI", id='calculate-roi-btn', n_clicks=0,
                               style={'width': '100%', 'padding': '15px', 'fontSize': '16px'})
                ], className="control-panel"),
            ]),
            
            html.Div([
                dcc.Graph(id='roi-cashflow-chart')
            ], style={'marginTop': '20px'}),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(id='roi-percentage', className="metric-value"),
                        html.Div("ROI Percentage", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='payback-period', className="metric-value"),
                        html.Div("Payback Period", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='annual-savings-roi', className="metric-value"),
                        html.Div("Annual Savings", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='net-profit', className="metric-value"),
                        html.Div("Net Profit", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
            ], className="row", style={'display': 'flex', 'justifyContent': 'space-between'})
        ], className="tab-container")
    
    elif tab == 'tab-4':
        return html.Div([
            html.H2("📈 Real-time Monitoring", style={'textAlign': 'center'}),
            
            html.Div([
                dcc.Graph(id='real-time-chart')
            ]),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div(id='current-value', className="metric-value"),
                        html.Div("Current Value", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='min-value', className="metric-value"),
                        html.Div("Min Today", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='max-value', className="metric-value"),
                        html.Div("Max Today", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
                
                html.Div([
                    html.Div([
                        html.Div(id='anomalies-count', className="metric-value"),
                        html.Div("Anomalies", className="metric-label")
                    ], className="metric-card"),
                ], className="three columns"),
            ], className="row", style={'display': 'flex', 'justifyContent': 'space-between'}),
            
            html.Div([
                html.Button("🔄 Reset Data", id='reset-data-btn', n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button("📥 Export Data", id='export-data-btn', n_clicks=0,
                           style={'margin': '10px', 'padding': '10px 20px'})
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], className="tab-container")

# ==================== 3D NETWORK CALLBACKS ====================
@app.callback(
    [Output('3d-network-graph', 'figure'),
     Output('sensor-count', 'children'),
     Output('coverage-percentage', 'children'),
     Output('optimal-spacing', 'children'),
     Output('building-size', 'children')],
    [Input('update-3d-btn', 'n_clicks'),
     Input('building-area-slider', 'value'),
     Input('sensor-count-slider', 'value')]
)
def update_3d_network(n_clicks, building_area, num_sensors):
    fig = create_3d_network(building_area, num_sensors)
    
    # Calculate metrics
    length = building_area ** 0.5 * 1.2
    width = building_area ** 0.5 * 0.8
    
    sensor_count = min(num_sensors, int((length-2)/5) * int((width-2)/5) * 2)
    coverage = min(100, (sensor_count * 25) / building_area * 100)
    
    return [
        fig,
        f"{sensor_count}",
        f"{coverage:.1f}%",
        "5.0 m",
        f"{building_area} m²"
    ]

# ==================== ENERGY ANALYTICS CALLBACKS ====================
@app.callback(
    [Output('energy-heatmap', 'figure'),
     Output('daily-consumption', 'children'),
     Output('daily-cost', 'children'),
     Output('monthly-savings', 'children'),
     Output('annual-savings-energy', 'children')],
    [Input('peak-load-slider', 'value'),
     Input('efficiency-slider', 'value')]
)
def update_energy_analytics(peak_adjustment, efficiency):
    # Generate heatmap data
    hourly_base = np.array([50, 45, 40, 38, 40, 60, 100, 150, 180, 170, 160, 155,
                            160, 165, 170, 180, 200, 220, 210, 180, 140, 100, 70, 55])
    
    hourly_data = hourly_base * (1 + peak_adjustment/100) * (efficiency/100)
    
    # Create heatmap
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [f'{h:02d}:00' for h in range(24)]
    
    # Add some variation between days
    data = []
    for day in range(7):
        variation = np.random.normal(1, 0.1, 24)
        data.append(hourly_data * variation)
    
    data = np.array(data)
    
    fig = px.imshow(
        data,
        x=hours,
        y=days,
        color_continuous_scale='Viridis',
        title="Weekly Energy Consumption Heatmap"
    )
    fig.update_layout(height=400)
    
    # Calculate metrics
    daily_consumption = np.sum(hourly_data)
    daily_cost = daily_consumption * 0.15
    monthly_savings = (np.sum(hourly_base) - daily_consumption) * 0.15 * 30
    annual_savings = monthly_savings * 12
    
    return [
        fig,
        f"{daily_consumption:.0f} kWh",
        f"${daily_cost:.2f}",
        f"${monthly_savings:.2f}",
        f"${annual_savings:.2f}"
    ]

# ==================== ROI CALCULATOR CALLBACKS ====================
@app.callback(
    [Output('roi-cashflow-chart', 'figure'),
     Output('roi-percentage', 'children'),
     Output('payback-period', 'children'),
     Output('annual-savings-roi', 'children'),
     Output('net-profit', 'children')],
    [Input('calculate-roi-btn', 'n_clicks')],
    [State('investment-input', 'value'),
     State('roi-area-input', 'value'),
     State('energy-cost-slider', 'value'),
     State('years-slider', 'value')]
)
def update_roi_calculator(n_clicks, investment, area, energy_cost, years):
    if n_clicks == 0:
        return [go.Figure(), "0%", "0 years", "$0", "$0"]
    
    results = calculate_roi(investment, area, energy_cost, years)
    
    # Create cash flow chart
    years_list = list(range(years + 1))
    cash_flows = [-investment] + [results['annual_savings']] * years
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years_list,
        y=cash_flows,
        name='Annual Cash Flow',
        marker_color=['red'] + ['green'] * years
    ))
    
    # Add cumulative line
    cumulative = np.cumsum(cash_flows)
    fig.add_trace(go.Scatter(
        x=years_list,
        y=cumulative,
        name='Cumulative',
        line=dict(color='blue', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Cash Flow Analysis',
        xaxis_title='Year',
        yaxis_title='Annual Cash Flow ($)',
        yaxis2=dict(
            title='Cumulative Cash Flow ($)',
            overlaying='y',
            side='right'
        ),
        height=400
    )
    
    return [
        fig,
        f"{results['roi']}%",
        f"{results['payback']} years",
        f"${results['annual_savings']:,.0f}",
        f"${results['net_profit']:,.0f}"
    ]

# ==================== REAL-TIME MONITORING CALLBACKS ====================
@app.callback(
    [Output('real-time-chart', 'figure'),
     Output('current-value', 'children'),
     Output('min-value', 'children'),
     Output('max-value', 'children'),
     Output('anomalies-count', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_real_time_monitor(n):
    # Generate real-time data
    np.random.seed(42)
    time_points = min(100, n % 200 + 50)
    time = np.linspace(0, 10, time_points)
    
    # Create sensor data with trend and noise
    trend = 50 + 10 * np.sin(time * 0.5 + n * 0.1)
    noise = np.random.normal(0, 2, time_points)
    sensor_data = trend + noise
    
    # Add occasional anomalies
    if n % 50 == 0 and time_points > 10:
        anomaly_idx = np.random.randint(5, time_points-5)
        sensor_data[anomaly_idx] += np.random.normal(20, 5)
    
    # Detect anomalies (simple threshold)
    mean_val = np.mean(sensor_data)
    std_val = np.std(sensor_data)
    anomalies = np.where(np.abs(sensor_data - mean_val) > 2.5 * std_val)[0]
    
    # Create chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(time_points)),
        y=sensor_data,
        mode='lines',
        name='Sensor Data',
        line=dict(color='blue', width=2)
    ))
    
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies,
            y=sensor_data[anomalies],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title='Real-time Sensor Monitoring',
        xaxis_title='Time Index',
        yaxis_title='Sensor Value',
        height=400,
        showlegend=True
    )
    
    return [
        fig,
        f"{sensor_data[-1]:.1f}",
        f"{np.min(sensor_data):.1f}",
        f"{np.max(sensor_data):.1f}",
        f"{len(anomalies)}"
    ]

# ==================== RUN THE APP ====================
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)

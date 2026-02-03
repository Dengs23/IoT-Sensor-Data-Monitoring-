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
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                margin: 5px;
            }
            .metric-value {
                font-size: 2rem;
                font-weight: bold;
                color: #1E3A8A;
            }
            .metric-label {
                font-size: 0.9rem;
                color: #666;
                margin-top: 5px;
            }
            .control-panel {
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }
            .row {
                display: flex;
                flex-wrap: wrap;
                margin: 10px -5px;
            }
            .three-columns {
                flex: 1;
                min-width: 200px;
                padding: 5px;
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
        ]),
        html.Div(id='tabs-content')
    ], className="main-container"),
    
    dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
])

# ==================== 3D VISUALIZATION ====================
def create_3d_network(building_area=1000, num_sensors=20):
    length = building_area ** 0.5 * 1.2
    width = building_area ** 0.5 * 0.8
    height = 4
    
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
    
    fig = go.Figure()
    
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
            marker=dict(size=6, color=coords[:, 2], colorscale='Viridis', opacity=0.8),
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
            html.H2("📊 3D Sensor Network", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Building Area (m²)"),
                    dcc.Slider(
                        id='building-area-slider',
                        min=100,
                        max=5000,
                        value=1000,
                        step=100
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Number of Sensors"),
                    dcc.Slider(
                        id='sensor-count-slider',
                        min=5,
                        max=100,
                        value=20,
                        step=5
                    ),
                ], className="control-panel"),
                
                html.Button("🔄 Update", id='update-3d-btn', n_clicks=0,
                           style={'width': '100%', 'padding': '10px', 'marginBottom': '20px'}),
            ]),
            
            dcc.Graph(id='3d-network-graph'),
            
            html.Div([
                html.Div([
                    html.Div(id='sensor-count', className="metric-value"),
                    html.Div("Sensors", className="metric-label")
                ], className="metric-card"),
            ], className="row")
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
                        step=5
                    ),
                ], className="control-panel"),
            ]),
            
            dcc.Graph(id='energy-heatmap'),
            
            html.Div([
                html.Div([
                    html.Div(id='daily-consumption', className="metric-value"),
                    html.Div("Daily (kWh)", className="metric-label")
                ], className="metric-card"),
                
                html.Div([
                    html.Div(id='daily-cost', className="metric-value"),
                    html.Div("Daily Cost", className="metric-label")
                ], className="metric-card"),
            ], className="row")
        ], className="tab-container")
    
    elif tab == 'tab-3':
        return html.Div([
            html.H2("💰 ROI Calculator", style={'textAlign': 'center'}),
            
            html.Div([
                html.Div([
                    html.Label("Investment ($)"),
                    dcc.Input(
                        id='investment-input',
                        type='number',
                        value=50000,
                        style={'width': '100%', 'padding': '10px'}
                    ),
                ], className="control-panel"),
                
                html.Div([
                    html.Label("Area (m²)"),
                    dcc.Input(
                        id='roi-area-input',
                        type='number',
                        value=2000,
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
                        step=0.01
                    ),
                ], className="control-panel"),
                
                html.Button("📊 Calculate", id='calculate-roi-btn', n_clicks=0,
                           style={'width': '100%', 'padding': '10px', 'marginBottom': '20px'}),
            ]),
            
            dcc.Graph(id='roi-cashflow-chart'),
            
            html.Div([
                html.Div([
                    html.Div(id='roi-percentage', className="metric-value"),
                    html.Div("ROI %", className="metric-label")
                ], className="metric-card"),
                
                html.Div([
                    html.Div(id='payback-period', className="metric-value"),
                    html.Div("Payback", className="metric-label")
                ], className="metric-card"),
            ], className="row")
        ], className="tab-container")

@app.callback(
    Output('3d-network-graph', 'figure'),
    [Input('update-3d-btn', 'n_clicks'),
     Input('building-area-slider', 'value'),
     Input('sensor-count-slider', 'value')]
)
def update_3d_network(n_clicks, building_area, num_sensors):
    return create_3d_network(building_area, num_sensors)

@app.callback(
    [Output('energy-heatmap', 'figure'),
     Output('daily-consumption', 'children'),
     Output('daily-cost', 'children')],
    [Input('peak-load-slider', 'value')]
)
def update_energy_analytics(peak_adjustment):
    hourly_base = np.array([50, 45, 40, 38, 40, 60, 100, 150, 180, 170, 160, 155,
                            160, 165, 170, 180, 200, 220, 210, 180, 140, 100, 70, 55])
    
    hourly_data = hourly_base * (1 + peak_adjustment/100)
    
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = [f'{h:02d}:00' for h in range(24)]
    
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
        title="Weekly Energy Consumption"
    )
    fig.update_layout(height=400)
    
    daily_consumption = np.sum(hourly_data)
    daily_cost = daily_consumption * 0.15
    
    return [
        fig,
        f"{daily_consumption:.0f}",
        f"${daily_cost:.2f}"
    ]

@app.callback(
    [Output('roi-cashflow-chart', 'figure'),
     Output('roi-percentage', 'children'),
     Output('payback-period', 'children')],
    [Input('calculate-roi-btn', 'n_clicks')],
    [State('investment-input', 'value'),
     State('roi-area-input', 'value'),
     State('energy-cost-slider', 'value')]
)
def update_roi_calculator(n_clicks, investment, area, energy_cost):
    years = 10
    results = calculate_roi(investment, area, energy_cost, years)
    
    years_list = list(range(years + 1))
    cash_flows = [-investment] + [results['annual_savings']] * years
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years_list,
        y=cash_flows,
        name='Cash Flow',
        marker_color=['red'] + ['green'] * years
    ))
    
    fig.update_layout(
        title='Cash Flow Analysis',
        xaxis_title='Year',
        yaxis_title='$',
        height=400
    )
    
    return [
        fig,
        f"{results['roi']}%",
        f"{results['payback']} yrs"
    ]

# ==================== RUN THE APP ====================
if __name__ == '__main__':
    app.run(debug=True, port=8050)

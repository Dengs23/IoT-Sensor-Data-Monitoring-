# iot_dash_complete_fixed.py
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "🏢 IoT Building Assistant"
server = app.server

# Building database
BUILDINGS = {
    "Empire State Building": {
        "location": "New York, USA",
        "area": 257211,
        "investment": 20000000,
        "savings": 38,
        "floors": 102,
        "height": 443
    },
    "Burj Khalifa": {
        "location": "Dubai, UAE",
        "area": 309473,
        "investment": 50000000,
        "savings": 23,
        "floors": 163,
        "height": 828
    },
    "Shanghai Tower": {
        "location": "Shanghai, China",
        "area": 420000,
        "investment": 30000000,
        "savings": 32,
        "floors": 128,
        "height": 632
    }
}

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏢 AI-Powered IoT Building Assistant"),
        html.P("Calculate ROI, analyze savings, and optimize building performance")
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # Main content
    html.Div([
        # Left panel - Inputs
        html.Div([
            html.H3("🔧 Configuration"),
            html.Label("Select Building:"),
            dcc.Dropdown(
                id='building-select',
                options=[{'label': k, 'value': k} for k in BUILDINGS.keys()],
                placeholder="Choose a building..."
            ),
            
            html.Div([
                html.Label("Or Enter Custom Values:"),
                html.Div([
                    html.Label("Area (m²):"),
                    dcc.Input(id='custom-area', type='number', value=1000, 
                             style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Investment ($):"),
                    dcc.Input(id='custom-investment', type='number', value=50000,
                             style={'width': '100%', 'marginBottom': '10px'})
                ]),
                html.Div([
                    html.Label("Energy Savings (%):"),
                    dcc.Slider(id='savings-slider', min=10, max=50, value=25, step=5,
                              marks={i: f'{i}%' for i in range(10, 51, 10)})
                ])
            ], style={'marginTop': '20px'}),
            
            html.Button('🚀 Calculate ROI', id='calculate-btn',
                       style={'width': '100%', 'marginTop': '20px', 'padding': '15px',
                              'backgroundColor': '#007bff', 'color': 'white',
                              'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'})
        ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                 'borderRadius': '10px', 'marginRight': '20px'}),
        
        # Right panel - Results
        html.Div([
            html.H3("📊 Results"),
            html.Div(id='results-container')
        ], style={'flex': '2'})
    ], style={'display': 'flex'}),
    
    # Charts section
    html.Div([
        dcc.Graph(id='roi-chart'),
        dcc.Graph(id='savings-chart')
    ], style={'marginTop': '30px'}),
    
    # Store for data
    dcc.Store(id='calculation-data')
])

# Callbacks
@app.callback(
    Output('calculation-data', 'data'),
    Input('calculate-btn', 'n_clicks'),
    [State('building-select', 'value'),
     State('custom-area', 'value'),
     State('custom-investment', 'value'),
     State('savings-slider', 'value')]
)
def calculate_roi(n_clicks, building_name, custom_area, custom_investment, savings_pct):
    if not n_clicks:
        return None
    
    # Use building data if selected, otherwise custom
    if building_name and building_name in BUILDINGS:
        building = BUILDINGS[building_name]
        area = building['area']
        investment = building['investment']
        savings_pct = building['savings']
        location = building['location']
    else:
        area = custom_area or 1000
        investment = custom_investment or 50000
        savings_pct = savings_pct or 25
        location = "Custom Building"
    
    # Calculations
    annual_energy_savings = investment * (savings_pct / 100)
    labor_savings = investment * 0.10  # 10% labor savings
    maintenance_savings = investment * 0.05  # 5% maintenance savings
    total_annual_savings = annual_energy_savings + labor_savings + maintenance_savings
    
    payback_years = investment / total_annual_savings if total_annual_savings > 0 else 0
    
    # 5-year projection
    years = list(range(1, 6))
    cumulative_savings = [total_annual_savings * year for year in years]
    cumulative_net = [savings - investment for savings in cumulative_savings]
    
    return {
        'building_name': building_name or 'Custom Building',
        'location': location,
        'area': area,
        'investment': investment,
        'savings_pct': savings_pct,
        'annual_savings': total_annual_savings,
        'payback_years': payback_years,
        'years': years,
        'cumulative_savings': cumulative_savings,
        'cumulative_net': cumulative_net,
        'breakdown': {
            'energy': annual_energy_savings,
            'labor': labor_savings,
            'maintenance': maintenance_savings
        }
    }

@app.callback(
    Output('results-container', 'children'),
    Input('calculation-data', 'data')
)
def display_results(data):
    if not data:
        return html.Div([
            html.P("Configure the parameters on the left and click 'Calculate ROI'"),
            html.P("The dashboard supports:"),
            html.Ul([
                html.Li("🏢 Famous buildings database"),
                html.Li("💰 Custom building configuration"),
                html.Li("📈 ROI calculations"),
                html.Li("📊 Interactive visualizations")
            ])
        ])
    
    return html.Div([
        html.H4(f"🏢 {data['building_name']}"),
        html.P(f"📍 {data['location']}"),
        
        html.Div([
            html.Div([
                html.H5("Building Area"),
                html.H3(f"{data['area']:,} m²")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H5("IoT Investment"),
                html.H3(f"${data['investment']:,}")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H5("Energy Savings"),
                html.H3(f"{data['savings_pct']}%")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
            
            html.Div([
                html.H5("Payback Period"),
                html.H3(f"{data['payback_years']:.1f} years")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px',
                  'margin': '20px 0'}),
        
        html.H5("Annual Savings Breakdown"),
        html.Div([
            html.Div([
                html.P("⚡ Energy Savings"),
                html.H4(f"${data['breakdown']['energy']:,.0f}")
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e3f2fd',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
            
            html.Div([
                html.P("👷 Labor Savings"),
                html.H4(f"${data['breakdown']['labor']:,.0f}")
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#f3e5f5',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
            
            html.Div([
                html.P("🔧 Maintenance Savings"),
                html.H4(f"${data['breakdown']['maintenance']:,.0f}")
            ], style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#e8f5e8',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'})
        ], style={'display': 'flex', 'marginBottom': '20px'})
    ])

@app.callback(
    Output('roi-chart', 'figure'),
    Input('calculation-data', 'data')
)
def update_roi_chart(data):
    if not data:
        return go.Figure()
    
    fig = go.Figure()
    
    # Cumulative savings line
    fig.add_trace(go.Scatter(
        x=data['years'],
        y=data['cumulative_savings'],
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='green', width=3)
    ))
    
    # Investment line (break-even)
    fig.add_trace(go.Scatter(
        x=data['years'],
        y=[data['investment']] * len(data['years']),
        mode='lines',
        name='Investment',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Net profit area
    fig.add_trace(go.Scatter(
        x=data['years'],
        y=data['cumulative_net'],
        fill='tozeroy',
        mode='lines',
        name='Net Profit',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="📈 5-Year ROI Projection",
        xaxis_title="Years",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

@app.callback(
    Output('savings-chart', 'figure'),
    Input('calculation-data', 'data')
)
def update_savings_chart(data):
    if not data:
        return px.pie(names=['No data'], values=[1], title="Savings Breakdown")
    
    breakdown = data['breakdown']
    categories = ['Energy', 'Labor', 'Maintenance']
    values = [breakdown['energy'], breakdown['labor'], breakdown['maintenance']]
    
    fig = px.pie(
        names=categories,
        values=values,
        title="💵 Annual Savings Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

# Run app - FIXED FOR NEW DASH VERSION
if __name__ == '__main__':
    app.run(debug=True, port=8050)  # Changed from app.run_server to app.run

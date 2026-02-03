import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import random
import os

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "🏢 AI-Powered IoT Building Assistant"
server = app.server

# ==================== COMPREHENSIVE BUILDING DATABASE ====================
FAMOUS_BUILDINGS = {
    "empire state building": {
        "name": "Empire State Building",
        "location": "New York, USA",
        "year_built": 1931,
        "height_m": 443.2,
        "floors": 102,
        "total_area_m2": 257211,
        "office_area_m2": 208879,
        "retail_area_m2": 24154,
        "annual_energy_use_kwh": 55000000,
        "annual_visitors": 4000000,
        "elevators": 73,
        "windows": 6514,
        "iot_investment_usd": 20000000,
        "energy_savings_percent": 38,
        "roi_months": 54,
        "description": "Iconic Art Deco skyscraper, underwent $550M retrofit including $20M IoT systems"
    },
    "burj khalifa": {
        "name": "Burj Khalifa",
        "location": "Dubai, UAE",
        "year_built": 2010,
        "height_m": 828,
        "floors": 163,
        "total_area_m2": 309473,
        "residential_area_m2": 104209,
        "office_area_m2": 33428,
        "hotel_area_m2": 13486,
        "annual_energy_use_kwh": 65000000,
        "annual_visitors": 1700000,
        "elevators": 57,
        "windows": 24400,
        "iot_investment_usd": 50000000,
        "energy_savings_percent": 23,
        "roi_months": 48,
        "description": "World's tallest building, LEED Gold certified, advanced building management system"
    },
    "shanghai tower": {
        "name": "Shanghai Tower",
        "location": "Shanghai, China",
        "year_built": 2015,
        "height_m": 632,
        "floors": 128,
        "total_area_m2": 420000,
        "office_area_m2": 212000,
        "hotel_area_m2": 88000,
        "retail_area_m2": 50000,
        "annual_energy_use_kwh": 48000000,
        "annual_visitors": 2500000,
        "elevators": 106,
        "windows": 20941,
        "iot_investment_usd": 30000000,
        "energy_savings_percent": 32,
        "roi_months": 42,
        "description": "Twisted skyscraper with double-skin facade, 270 wind turbines, LEED Platinum"
    },
    "the gherkin": {
        "name": "30 St Mary Axe (The Gherkin)",
        "location": "London, UK",
        "year_built": 2003,
        "height_m": 180,
        "floors": 41,
        "total_area_m2": 47600,
        "office_area_m2": 46500,
        "annual_energy_use_kwh": 8500000,
        "annual_visitors": 100000,
        "elevators": 23,
        "windows": 5390,
        "iot_investment_usd": 8000000,
        "energy_savings_percent": 30,
        "roi_months": 38,
        "description": "Swiss Re headquarters, energy-efficient cigar shape, natural ventilation"
    },
    "marina bay sands": {
        "name": "Marina Bay Sands",
        "location": "Singapore",
        "year_built": 2010,
        "height_m": 200,
        "floors": 55,
        "total_area_m2": 845000,
        "hotel_rooms": 2561,
        "casino_area_m2": 15000,
        "retail_area_m2": 93000,
        "annual_energy_use_kwh": 350000000,
        "annual_visitors": 50000000,
        "elevators": 141,
        "iot_investment_usd": 75000000,
        "energy_savings_percent": 28,
        "roi_months": 60,
        "description": "Integrated resort with hotel, casino, mall, museum, and Skypark"
    }
}

# ==================== 3D VISUALIZATION ====================
def create_3d_building(length=50, width=30, height=20):
    """Create 3D building visualization"""
    fig = go.Figure()
    
    # Building wireframe
    x_box = [0, length, length, 0, 0, length, length, 0]
    y_box = [0, 0, width, width, 0, 0, width, width]
    z_box = [0, 0, 0, 0, height, height, height, height]
    
    fig.add_trace(go.Mesh3d(
        x=x_box, y=y_box, z=z_box,
        color='lightblue', opacity=0.1, name='Building'
    ))
    
    # Add sensors
    sensors = []
    for x in np.arange(5, length, 10):
        for y in np.arange(5, width, 10):
            for z in [2, height-2]:
                sensors.append([x, y, z])
    
    if sensors:
        sensors = np.array(sensors)
        fig.add_trace(go.Scatter3d(
            x=sensors[:, 0], y=sensors[:, 1], z=sensors[:, 2],
            mode='markers', marker=dict(size=5, color='red', opacity=0.8),
            name='IoT Sensors'
        ))
    
    fig.update_layout(
        title="🏗️ 3D Building Model with IoT Sensors",
        scene=dict(
            xaxis_title='Length (m)', yaxis_title='Width (m)', zaxis_title='Height (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

# ==================== APP LAYOUT ====================
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏢 AI-Powered IoT Building Assistant", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.H3("Complete Analysis: Recognition • ROI • 3D Visualization • Sensor Planning",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
        html.P("iot_assistant.co.za", 
               style={'textAlign': 'center', 'color': '#3498db', 'fontWeight': 'bold', 'marginBottom': '10px'})
    ], className='header'),
    
    # Tabs for different features
    dcc.Tabs(id='main-tabs', value='tab-recognition', children=[
        # Tab 1: Building Recognition
        dcc.Tab(label='🤖 AI Recognition', value='tab-recognition', children=[
            html.Div([
                html.H3("🏛️ Building Recognition Assistant"),
                html.P("Describe any building for instant analysis:"),
                
                html.Div([
                    dcc.Textarea(
                        id='building-input',
                        placeholder="e.g., 'Empire State Building' or 'large office tower in New York with 100 floors'",
                        style={'width': '100%', 'height': '100px', 'padding': '15px', 'fontSize': '16px',
                               'borderRadius': '10px', 'border': '2px solid #dee2e6', 'marginBottom': '20px'}
                    ),
                    
                    html.Div([
                        html.Button('🔍 AI Recognition', id='ai-recognize-btn',
                                   style={'padding': '12px 30px', 'marginRight': '10px',
                                          'backgroundColor': '#10b981', 'color': 'white', 'border': 'none',
                                          'borderRadius': '5px', 'cursor': 'pointer'}),
                        html.Button('📊 Quick Select', id='quick-select-btn',
                                   style={'padding': '12px 30px',
                                          'backgroundColor': '#3b82f6', 'color': 'white', 'border': 'none',
                                          'borderRadius': '5px', 'cursor': 'pointer'})
                    ], style={'marginBottom': '30px'})
                ]),
                
                # Recognition Results
                html.Div(id='recognition-results', style={'marginBottom': '30px'}),
                
                # Famous Buildings Grid
                html.H4("🏛️ Famous Buildings Database"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H5("🏢 Empire State"),
                            html.P("New York, USA"),
                            html.P("257,211 m²"),
                            html.P("$20M IoT")
                        ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                                 'cursor': 'pointer'}, id='btn-empire')
                    ], style={'flex': '1', 'margin': '5px'}),
                    
                    html.Div([
                        html.Div([
                            html.H5("🏢 Burj Khalifa"),
                            html.P("Dubai, UAE"),
                            html.P("309,473 m²"),
                            html.P("$50M IoT")
                        ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                                 'cursor': 'pointer'}, id='btn-burj')
                    ], style={'flex': '1', 'margin': '5px'}),
                    
                    html.Div([
                        html.Div([
                            html.H5("🏢 Shanghai Tower"),
                            html.P("Shanghai, China"),
                            html.P("420,000 m²"),
                            html.P("$30M IoT")
                        ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                                 'cursor': 'pointer'}, id='btn-shanghai')
                    ], style={'flex': '1', 'margin': '5px'}),
                    
                    html.Div([
                        html.Div([
                            html.H5("🏢 The Gherkin"),
                            html.P("London, UK"),
                            html.P("47,600 m²"),
                            html.P("$8M IoT")
                        ], style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                                 'cursor': 'pointer'}, id='btn-gherkin')
                    ], style={'flex': '1', 'margin': '5px'})
                ], style={'display': 'flex', 'marginBottom': '30px'})
            ], style={'padding': '30px'})
        ]),
        
        # Tab 2: ROI Calculator
        dcc.Tab(label='💰 ROI Calculator', value='tab-roi', children=[
            html.Div([
                html.H3("📈 Advanced ROI Calculator"),
                
                html.Div([
                    # Left Column
                    html.Div([
                        html.H5("Building Parameters"),
                        html.Label("Building Area (m²):"),
                        dcc.Input(id='area-input', type='number', value=1000,
                                 style={'width': '100%', 'padding': '10px', 'marginBottom': '15px'}),
                        
                        html.Label("IoT Investment ($):"),
                        dcc.Input(id='investment-input', type='number', value=50000,
                                 style={'width': '100%', 'padding': '10px', 'marginBottom': '15px'}),
                        
                        html.Label("Energy Cost ($/kWh):"),
                        dcc.Slider(id='energy-slider', min=0.05, max=0.50, value=0.15, step=0.01,
                                  marks={0.05: '$0.05', 0.15: '$0.15', 0.30: '$0.30', 0.50: '$0.50'},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Building Type:"),
                        dcc.Dropdown(
                            id='building-type',
                            options=[
                                {'label': '🏢 Office', 'value': 'office'},
                                {'label': '🛍️ Retail', 'value': 'retail'},
                                {'label': '🏨 Hotel', 'value': 'hotel'},
                                {'label': '🏥 Hospital', 'value': 'hospital'},
                                {'label': '🏟️ Stadium', 'value': 'stadium'}
                            ],
                            value='office',
                            style={'marginBottom': '20px'}
                        )
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                             'borderRadius': '10px', 'marginRight': '20px'}),
                    
                    # Right Column
                    html.Div([
                        html.H5("Additional Parameters"),
                        html.Label("Annual Labor Cost ($):"),
                        dcc.Input(id='labor-input', type='number', value=50000,
                                 style={'width': '100%', 'padding': '10px', 'marginBottom': '15px'}),
                        
                        html.Label("Annual Maintenance Cost ($):"),
                        dcc.Input(id='maintenance-input', type='number', value=25000,
                                 style={'width': '100%', 'padding': '10px', 'marginBottom': '15px'}),
                        
                        html.Label("Analysis Period (Years):"),
                        dcc.Slider(id='years-slider', min=1, max=20, value=10, step=1,
                                  marks={1: '1', 5: '5', 10: '10', 15: '15', 20: '20'},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Button('📊 Calculate Comprehensive ROI', id='calculate-roi-btn',
                                   style={'width': '100%', 'padding': '15px', 'marginTop': '20px',
                                          'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                                          'borderRadius': '5px', 'cursor': 'pointer'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                             'borderRadius': '10px'})
                ], style={'display': 'flex', 'marginBottom': '30px'}),
                
                # Results Section
                html.Div(id='roi-results')
            ], style={'padding': '30px'})
        ]),
        
        # Tab 3: 3D Visualization
        dcc.Tab(label='🏗️ 3D Visualization', value='tab-3d', children=[
            html.Div([
                html.H3("📊 3D Building Visualization"),
                html.P("Interactive 3D model with IoT sensor placement"),
                
                dcc.Graph(id='3d-building-viz', figure=create_3d_building()),
                
                html.Div([
                    html.Div([
                        html.Label("Building Length (m):"),
                        dcc.Slider(id='length-slider', min=20, max=100, value=50, step=5,
                                  marks={20: '20m', 50: '50m', 80: '80m', 100: '100m'})
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Building Width (m):"),
                        dcc.Slider(id='width-slider', min=15, max=80, value=30, step=5,
                                  marks={15: '15m', 30: '30m', 50: '50m', 80: '80m'})
                    ], style={'flex': '1', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.Label("Building Height (m):"),
                        dcc.Slider(id='height-slider', min=10, max=100, value=20, step=5,
                                  marks={10: '10m', 30: '30m', 50: '50m', 80: '80m', 100: '100m'})
                    ], style={'flex': '1'})
                ], style={'display': 'flex', 'marginTop': '30px'}),
                
                html.Div([
                    html.Button('🔄 Update 3D Model', id='update-3d-btn',
                               style={'padding': '12px 30px', 'marginTop': '20px',
                                      'backgroundColor': '#8b5cf6', 'color': 'white', 'border': 'none',
                                      'borderRadius': '5px', 'cursor': 'pointer'})
                ])
            ], style={'padding': '30px'})
        ]),
        
        # Tab 4: Sensor Recommendations
        dcc.Tab(label='📡 Sensor Planning', value='tab-sensors', children=[
            html.Div([
                html.H3("🎯 IoT Sensor Deployment Plan"),
                
                html.Div([
                    html.Div([
                        html.H5("Sensor Configuration"),
                        html.Label("Sensor Type:"),
                        dcc.Dropdown(
                            id='sensor-type',
                            options=[
                                {'label': '🌡️ Temperature/Humidity', 'value': 'temp'},
                                {'label': '👥 Occupancy', 'value': 'occupancy'},
                                {'label': '💡 Lighting', 'value': 'lighting'},
                                {'label': '⚡ Energy', 'value': 'energy'},
                                {'label': '🔒 Security', 'value': 'security'},
                                {'label': '💧 Water Flow', 'value': 'water'}
                            ],
                            value=['temp', 'occupancy'],
                            multi=True,
                            style={'marginBottom': '20px'}
                        ),
                        
                        html.Label("Sensor Density:"),
                        dcc.Slider(id='density-slider', min=1, max=10, value=5, step=1,
                                  marks={1: 'Low', 5: 'Medium', 10: 'High'},
                                  tooltip={"placement": "bottom", "always_visible": True}),
                        
                        html.Label("Budget Range:"),
                        dcc.RadioItems(
                            id='budget-radio',
                            options=[
                                {'label': '💰 Basic ($10-50k)', 'value': 'basic'},
                                {'label': '💵 Standard ($50-200k)', 'value': 'standard'},
                                {'label': '💎 Premium ($200k+)', 'value': 'premium'}
                            ],
                            value='standard',
                            style={'marginBottom': '20px'}
                        )
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                             'borderRadius': '10px', 'marginRight': '20px'}),
                    
                    html.Div([
                        html.H5("Deployment Schedule"),
                        html.Label("Phase Duration (weeks):"),
                        dcc.Slider(id='phase-slider', min=1, max=12, value=4, step=1,
                                  marks={1: '1w', 4: '4w', 8: '8w', 12: '12w'}),
                        
                        html.Label("Priority Areas:"),
                        dcc.Checklist(
                            id='priority-checklist',
                            options=[
                                {'label': '🏢 Office Spaces', 'value': 'office'},
                                {'label': '🛗 Elevators', 'value': 'elevators'},
                                {'label': '🛗 HVAC Systems', 'value': 'hvac'},
                                {'label': '💡 Lighting Zones', 'value': 'lighting'},
                                {'label': '🚪 Entry/Exit Points', 'value': 'entry'}
                            ],
                            value=['office', 'hvac'],
                            style={'marginBottom': '20px'}
                        ),
                        
                        html.Button('📋 Generate Deployment Plan', id='generate-plan-btn',
                                   style={'width': '100%', 'padding': '15px', 'marginTop': '20px',
                                          'backgroundColor': '#10b981', 'color': 'white', 'border': 'none',
                                          'borderRadius': '5px', 'cursor': 'pointer'})
                    ], style={'flex': '1', 'padding': '20px', 'backgroundColor': '#f8f9fa',
                             'borderRadius': '10px'})
                ], style={'display': 'flex', 'marginBottom': '30px'}),
                
                # Plan Results
                html.Div(id='sensor-plan-results')
            ], style={'padding': '30px'})
        ])
    ]),
    
    # Data stores
    dcc.Store(id='current-building-data'),
    dcc.Store(id='roi-calculation-data'),
    
    # Footer
    html.Div([
        html.Hr(),
        html.P("🏢 IoT Building Assistant • iot_assistant.co.za • Powered by Dash • Real-time Analytics",
              style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
    ], style={'marginTop': '50px', 'padding': '20px'})
])

# ==================== CALLBACKS ====================
@app.callback(
    Output('recognition-results', 'children'),
    [Input('ai-recognize-btn', 'n_clicks'),
     Input('quick-select-btn', 'n_clicks'),
     Input('btn-empire', 'n_clicks'),
     Input('btn-burj', 'n_clicks'),
     Input('btn-shanghai', 'n_clicks'),
     Input('btn-gherkin', 'n_clicks')],
    [State('building-input', 'value')]
)
def handle_recognition(ai_clicks, quick_clicks, *args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.P("Enter a building description or select from famous buildings.")
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    building_input = args[-1] if args else None
    
    if button_id == 'ai-recognize-btn' and building_input:
        # Simple recognition logic
        for key, data in FAMOUS_BUILDINGS.items():
            if key in building_input.lower():
                return display_building_info(data)
        return html.Div([
            html.H4("🔍 Building Analysis"),
            html.P(f"Analysis of: '{building_input}'"),
            html.P("💡 Not in famous buildings database. Using custom analysis."),
            html.P("Suggested parameters: Office building, 25% energy savings, 3-year payback")
        ])
    
    elif button_id == 'quick-select-btn' and building_input:
        return html.Div([
            html.H4("⚡ Quick Analysis"),
            html.P(f"Building: {building_input}"),
            html.P("📍 Custom location"),
            html.P("📊 Estimated area: 10,000 m²"),
            html.P("💰 Suggested IoT investment: $250,000")
        ])
    
    elif button_id.startswith('btn-'):
        building_map = {
            'btn-empire': 'empire state building',
            'btn-burj': 'burj khalifa',
            'btn-shanghai': 'shanghai tower',
            'btn-gherkin': 'the gherkin'
        }
        if button_id in building_map:
            return display_building_info(FAMOUS_BUILDINGS[building_map[button_id]])
    
    return html.P("Click a button to analyze a building.")

def display_building_info(building):
    return html.Div([
        html.H4(f"🏢 {building['name']}"),
        html.P(f"📍 {building['location']} • 📅 Built: {building['year_built']}"),
        html.P(f"📏 Height: {building['height_m']}m • 🏢 Floors: {building['floors']}"),
        html.P(f"📊 Total Area: {building['total_area_m2']:,} m²"),
        html.P(f"💰 IoT Investment: ${building['iot_investment_usd']:,}"),
        html.P(f"⚡ Energy Savings: {building['energy_savings_percent']}% • 📈 ROI: {building['roi_months']} months"),
        html.P(building['description'], style={'fontStyle': 'italic', 'marginTop': '10px',
                                              'backgroundColor': '#f8f9fa', 'padding': '15px',
                                              'borderRadius': '5px'})
    ])

@app.callback(
    Output('roi-results', 'children'),
    Input('calculate-roi-btn', 'n_clicks'),
    [State('area-input', 'value'),
     State('investment-input', 'value'),
     State('energy-slider', 'value'),
     State('building-type', 'value'),
     State('labor-input', 'value'),
     State('maintenance-input', 'value'),
     State('years-slider', 'value')]
)
def calculate_roi(n_clicks, area, investment, energy_cost, building_type, labor, maintenance, years):
    if not n_clicks:
        return html.P("Configure parameters and click 'Calculate Comprehensive ROI'")
    
    # Calculations
    base_energy = area * 200  # kWh per m²
    energy_savings = base_energy * energy_cost * 0.25  # 25% savings
    labor_savings = labor * 0.30  # 30% labor savings
    maintenance_savings = maintenance * 0.25  # 25% maintenance savings
    productivity = area * 2.0  # $2 per m² productivity gain
    
    total_annual = energy_savings + labor_savings + maintenance_savings + productivity
    total_savings = total_annual * years
    net_profit = total_savings - investment
    roi_pct = (net_profit / investment * 100) if investment > 0 else 0
    payback = investment / total_annual if total_annual > 0 else 0
    
    # Create charts
    years_list = list(range(1, years + 1))
    cumulative = [total_annual * y for y in years_list]
    
    roi_fig = go.Figure()
    roi_fig.add_trace(go.Bar(x=years_list, y=[total_annual]*years, name='Annual Savings'))
    roi_fig.add_trace(go.Scatter(x=years_list, y=cumulative, name='Cumulative', yaxis='y2',
                                line=dict(color='blue', width=3)))
    roi_fig.update_layout(
        title=f"📅 {years}-Year Savings Projection",
        xaxis_title="Year",
        yaxis_title="Annual Savings ($)",
        yaxis2=dict(title="Cumulative ($)", overlaying="y", side="right"),
        height=400
    )
    
    breakdown_fig = px.pie(
        names=['Energy', 'Labor', 'Maintenance', 'Productivity'],
        values=[energy_savings, labor_savings, maintenance_savings, productivity],
        title="💵 Savings Breakdown",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    return html.Div([
        html.H4("📊 ROI Analysis Results"),
        
        html.Div([
            html.Div([
                html.H5("ROI Percentage"),
                html.H2(f"{roi_pct:.1f}%", style={'color': 'green' if roi_pct > 0 else 'red'})
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '5px'}),
            
            html.Div([
                html.H5("Payback Period"),
                html.H2(f"{payback:.1f} years")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '5px'}),
            
            html.Div([
                html.H5("Annual Savings"),
                html.H2(f"${total_annual:,.0f}")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '5px'}),
            
            html.Div([
                html.H5("Total Savings"),
                html.H2(f"${total_savings:,.0f}")
            ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'white',
                     'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'margin': '5px'})
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '10px',
                  'marginBottom': '30px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(figure=roi_fig)
            ], style={'flex': '1', 'padding': '10px'}),
            
            html.Div([
                dcc.Graph(figure=breakdown_fig)
            ], style={'flex': '1', 'padding': '10px'})
        ], style={'display': 'flex'})
    ])

@app.callback(
    Output('3d-building-viz', 'figure'),
    [Input('update-3d-btn', 'n_clicks')],
    [State('length-slider', 'value'),
     State('width-slider', 'value'),
     State('height-slider', 'value')]
)
def update_3d_model(n_clicks, length, width, height):
    return create_3d_building(length, width, height)

@app.callback(
    Output('sensor-plan-results', 'children'),
    Input('generate-plan-btn', 'n_clicks'),
    [State('sensor-type', 'value'),
     State('density-slider', 'value'),
     State('budget-radio', 'value'),
     State('phase-slider', 'value'),
     State('priority-checklist', 'value')]
)
def generate_sensor_plan(n_clicks, sensor_types, density, budget, weeks, priorities):
    if not n_clicks:
        return html.P("Configure sensor parameters and click 'Generate Deployment Plan'")
    
    sensor_names = {
        'temp': 'Temperature/Humidity',
        'occupancy': 'Occupancy',
        'lighting': 'Smart Lighting',
        'energy': 'Energy Monitoring',
        'security': 'Security',
        'water': 'Water Flow'
    }
    
    priority_names = {
        'office': 'Office Spaces',
        'elevators': 'Elevators',
        'hvac': 'HVAC Systems',
        'lighting': 'Lighting Zones',
        'entry': 'Entry/Exit Points'
    }
    
    selected_sensors = [sensor_names.get(t, t) for t in sensor_types]
    selected_priorities = [priority_names.get(p, p) for p in priorities]
    
    budget_ranges = {
        'basic': '$10,000 - $50,000',
        'standard': '$50,000 - $200,000',
        'premium': '$200,000+'
    }
    
    sensor_count = density * 50
    estimated_cost = {
        'basic': sensor_count * 100,
        'standard': sensor_count * 300,
        'premium': sensor_count * 800
    }.get(budget, sensor_count * 300)
    
    return html.Div([
        html.H4("📡 IoT Sensor Deployment Plan"),
        
        html.Div([
            html.Div([
                html.H5("📋 Plan Summary"),
                html.P(f"🎯 Sensor Types: {', '.join(selected_sensors)}"),
                html.P(f"📊 Density Level: {density}/10"),
                html.P(f"💰 Budget Range: {budget_ranges.get(budget, budget)}"),
                html.P(f"⏱️ Deployment Time: {weeks} weeks"),
                html.P(f"🎯 Priority Areas: {', '.join(selected_priorities)}"),
                html.P(f"📱 Estimated Sensors: {sensor_count} units"),
                html.P(f"💵 Estimated Cost: ${estimated_cost:,.0f}")
            ], style={'padding': '20px', 'backgroundColor': '#f0f9ff', 'borderRadius': '10px',
                     'marginBottom': '20px'}),
            
            html.H5("📅 Deployment Phases"),
            html.Ul([
                html.Li(f"Phase 1 (Week 1-{max(1, weeks//4)}): Site assessment and planning"),
                html.Li(f"Phase 2 (Week {max(2, weeks//4)+1}-{weeks//2}): Core infrastructure"),
                html.Li(f"Phase 3 (Week {weeks//2+1}-{3*weeks//4}): Sensor deployment"),
                html.Li(f"Phase 4 (Week {3*weeks//4+1}-{weeks}): Testing and commissioning")
            ]),
            
            html.H5("✅ Key Recommendations"),
            html.Ul([
                html.Li("Start with high-traffic areas for maximum impact"),
                html.Li("Implement phased rollout to minimize disruption"),
                html.Li("Train maintenance staff on new systems"),
                html.Li("Establish baseline metrics before deployment"),
                html.Li("Plan for ongoing maintenance and updates")
            ])
        ])
    ])

# ==================== RUN APP ====================
if __name__ == '__main__':
    print("=" * 60)
    print("🏢 AI-Powered IoT Building Assistant")
    print("🌐 Domain: iot_assistant.co.za")
    print("=" * 60)
    print("\n🚀 Starting server...")
    print("📊 Dashboard will be available at: http://localhost:8050")
    print("👀 Check the interface footer for: iot_assistant.co.za")
    print("\nℹ️  Press CTRL+C to stop the server")
    print("=" * 60)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8050)

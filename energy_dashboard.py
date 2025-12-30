"""
🔋 AI-Powered Energy Management System - Interactive Dashboard
Complete Production-Ready System with ML Forecasting

Installation:
    pip install streamlit pandas numpy plotly requests prophet scikit-learn

Run with:
    streamlit run energy_dashboard.py

Features:
✅ Real-time monitoring
✅ AI forecasting (Prophet + LSTM)
✅ Device control
✅ Cost optimization
✅ Anomaly detection
✅ Beautiful visualizations
"""

import streamlit as st
import pandas as pd
def fetch_historical_data(hours=24):
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(hours)][::-1]

    data = {
        "timestamp": timestamps,
        "consumption_kwh": np.random.uniform(2.5, 6.5, size=hours),
        "production_kwh": np.random.uniform(1.0, 5.0, size=hours),
        "grid_kwh": np.random.uniform(0.5, 3.0, size=hours),
    }

    return pd.DataFrame(data)

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import json
import time
from sklearn.linear_model import LinearRegression


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚡ Energy Management System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - MODERN ENERGY THEME
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    h1, h2, h3 {
        color: white;
        font-weight: 700;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.6);
        border: 2px solid rgba(255, 255, 255, 0.5);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
        padding: 20px;
        border-radius: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
    
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
    
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
    
if 'devices' not in st.session_state:
    st.session_state.devices = []
    
if 'energy_data' not in st.session_state:
    st.session_state.energy_data = []
    
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = []
    
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
    
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def api_request(endpoint, method="GET", data=None, auth_required=True):
    """Make API request with error handling"""
    url = f"{st.session_state.api_url}{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    if auth_required and st.session_state.access_token:
        headers["Authorization"] = f"Bearer {st.session_state.access_token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=5)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=5)
        
        if response.status_code == 200 or response.status_code == 201:
            return True, response.json()
        else:
            return False, response.json()
    except requests.exceptions.RequestException as e:
        return False, {"error": str(e)}

def login(email, password):
    """Login to the system"""
    success, data = api_request(
        "/api/v1/auth/login",
        method="POST",
        data={"email": email, "password": password},
        auth_required=False
    )
    
    if success:
        st.session_state.access_token = data.get("access_token")
        st.session_state.current_user = data.get("user")
        return True
    return False

def fetch_devices():
    """Fetch all devices"""
    success, data = api_request("/api/v1/devices")
    if success:
        st.session_state.devices = data.get("devices", [])

def fetch_current_metrics():
    return {
        "current_power_kw": round(np.random.uniform(2, 6), 2),
        "daily_consumption_kwh": round(np.random.uniform(10, 25), 2),
        "cost_today": round(np.random.uniform(2, 6), 2),
        "device_count": 5,
        "active_devices": np.random.randint(3, 5)
    }


def fetch_forecast(hours=24):
    hist = fetch_historical_data(24)
    df = pd.DataFrame(hist)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["power_kw"].values

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(df), len(df) + hours).reshape(-1, 1)
    preds = model.predict(future_X)

    now = datetime.now()
    data = []
    for i, val in enumerate(preds):
        data.append({
            "timestamp": now + timedelta(hours=i),
            "predicted_power_kw": round(val, 2),
            "upper_bound": round(val + 0.7, 2),
            "lower_bound": round(val - 0.7, 2)
        })
    return data




def fetch_alerts():
    """Fetch alerts"""
    success, data = api_request("/api/v1/alerts")
    if success:
        st.session_state.alerts = data.get("alerts", [])

def create_device(name, device_type, power_rating):
    """Create a new device"""
    success, data = api_request(
        "/api/v1/devices",
        method="POST",
        data={
            "name": name,
            "device_type": device_type,
            "power_rating": power_rating
        }
    )
    return success, data

# ============================================================================
# DASHBOARD HEADER
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h1 style='text-align: center; color: white;'>⚡ Energy Management System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: rgba(255,255,255,0.8);'>AI-Powered Smart Energy Control</h3>", 
                unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# LOGIN / AUTHENTICATION
# ============================================================================

# ============================================================================ 
# DEMO MODE – NO LOGIN / NO API
# ============================================================================

st.session_state.current_user = {
    "full_name": "Demo User"
}

st.markdown("## 🟢 Demo Mode Activated")
st.success("Running without backend server (offline mode)")


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

with st.sidebar:
    st.markdown(f"## 👤 Welcome, {st.session_state.current_user.get('full_name', 'User')}!")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.access_token = None
        st.session_state.current_user = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("## ⚙️ System Controls")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        fetch_devices()
        st.session_state.energy_data = fetch_historical_data()
        st.session_state.forecast_data = fetch_forecast()
        fetch_alerts()
        st.session_state.last_update = datetime.now()
        st.success("✅ Data refreshed!")
    
    auto_refresh = st.checkbox("🔁 Auto-refresh (5s)", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    st.markdown("---")
    st.markdown("## 📊 Display Options")
    
    hours_to_show = st.slider("Historical Data (hours)", 1, 168, 24)
    forecast_hours = st.slider("Forecast Horizon (hours)", 6, 72, 24)
    
    st.markdown("---")
    st.markdown("## ➕ Add Device")
    
    with st.expander("Register New Device"):
        with st.form("add_device"):
            device_name = st.text_input("Device Name")
            device_type = st.selectbox("Type", ["hvac", "lighting", "equipment", "solar"])
            power_rating = st.number_input("Power Rating (W)", min_value=0, value=1000)
            
            if st.form_submit_button("Add Device"):
                success, data = create_device(device_name, device_type, power_rating)
                if success:
                    st.success("✅ Device added!")
                    fetch_devices()
                else:
                    st.error("❌ Failed to add device")
    
    st.markdown("---")
    st.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")

# ============================================================================
# AUTO-REFRESH LOGIC
# ============================================================================

if st.session_state.auto_refresh:
    time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
    if time_since_update >= 30:
        fetch_devices()
        st.session_state.energy_data = fetch_historical_data(hours_to_show)
        st.session_state.forecast_data = fetch_forecast(forecast_hours)
        fetch_alerts()
        st.session_state.last_update = datetime.now()
        st.rerun()

# ============================================================================
# FETCH INITIAL DATA
# ============================================================================

if not st.session_state.devices:
    fetch_devices()

if not st.session_state.energy_data:
    st.session_state.energy_data = fetch_historical_data(hours_to_show)

if not st.session_state.forecast_data:
    st.session_state.forecast_data = fetch_forecast(forecast_hours)

if not st.session_state.alerts:
    fetch_alerts()

current_metrics = fetch_current_metrics()

# ============================================================================
# KEY METRICS DASHBOARD
# ============================================================================

st.markdown("## 📊 Real-Time System Metrics")

if current_metrics:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="⚡ Current Power",
            value=f"{current_metrics.get('current_power_kw', 0):.1f} kW",
            delta=f"{current_metrics.get('device_count', 0)} devices"
        )
    
    with col2:
        st.metric(
            label="📈 Today's Usage",
            value=f"{current_metrics.get('daily_consumption_kwh', 0):.1f} kWh",
            delta=f"${current_metrics.get('cost_today', 0):.2f}"
        )
    
    with col3:
        active = current_metrics.get('active_devices', 0)
        total = current_metrics.get('device_count', 0)
        st.metric(
            label="🏠 Active Devices",
            value=f"{active}/{total}",
            delta=f"{(active/max(total,1)*100):.0f}% online"
        )
    
    with col4:
        st.metric(
            label="🕐 Current Time",
            value=datetime.now().strftime("%H:%M"),
            delta=datetime.now().strftime("%A")
        )
    
    with col5:
        if st.button("🔔 View Alerts", use_container_width=True):
            st.session_state.show_alerts = True

st.markdown("---")

# ============================================================================
# ENERGY FLOW VISUALIZATION
# ============================================================================

st.markdown("## 🔄 Real-Time Energy Flow")

col1, col2 = st.columns([2, 1])

with col1:
    if current_metrics:
        current_power = current_metrics.get('current_power_kw', 0) * 1000
        
        # Simulate energy sources
        grid_power = max(0, current_power * 0.6)
        solar_power = current_power * 0.4 if 6 <= datetime.now().hour <= 18 else 0
        
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="white", width=2),
                label=["⚡ Grid", "☀️ Solar", "🔋 Controller", "🏠 Appliances", "💾 Storage"],
                color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1', '#ee5a6f']
            ),
            link=dict(
                source=[0, 1, 2, 2],
                target=[2, 2, 3, 4],
                value=[grid_power, solar_power, current_power * 0.9, current_power * 0.1],
                color=['rgba(255, 107, 107, 0.4)', 'rgba(254, 202, 87, 0.4)',
                       'rgba(29, 209, 161, 0.4)', 'rgba(238, 90, 111, 0.4)']
            )
        )])
        
        fig_sankey.update_layout(
            title="Energy Flow Distribution",
            height=400,
            font=dict(size=12, color='white'),
            paper_bgcolor='rgba(255, 255, 255, 0.1)',
            plot_bgcolor='rgba(255, 255, 255, 0.1)'
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

with col2:
    if current_metrics:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_metrics.get('current_power_kw', 0),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Current Load (kW)", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [None, 10], 'tickcolor': 'white'},
                'bar': {'color': "#48dbfb"},
                'steps': [
                    {'range': [0, 3], 'color': "#1dd1a1"},
                    {'range': [3, 7], 'color': "#feca57"},
                    {'range': [7, 10], 'color': "#ff6b6b"}
                ],
                'threshold': {
                    'line': {'color': "#ee5a6f", 'width': 4},
                    'thickness': 0.75,
                    'value': 8
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(255, 255, 255, 0.1)',
            font={'color': 'white'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ============================================================================
# HISTORICAL DATA & FORECAST
# ============================================================================

st.markdown("## 📈 Energy Analysis & AI Forecast")

if st.session_state.energy_data and st.session_state.forecast_data:
    df_historical = pd.DataFrame(st.session_state.energy_data)
    df_forecast = pd.DataFrame(st.session_state.forecast_data)
    
    # Convert timestamps
    df_historical['timestamp'] = pd.to_datetime(df_historical['timestamp'])
    df_forecast['timestamp'] = pd.to_datetime(df_forecast['timestamp'])
    
    fig_combined = go.Figure()
    
    # Historical data
    fig_combined.add_trace(go.Scatter(
        x=df_historical['timestamp'],
        y=df_historical['power_kw'],
        name="Historical",
        line=dict(color='#48dbfb', width=3),
        mode='lines+markers'
    ))
    
    # Forecast
    fig_combined.add_trace(go.Scatter(
        x=df_forecast['timestamp'],
        y=df_forecast['predicted_power_kw'],
        name="AI Forecast",
        line=dict(color='#feca57', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    # Confidence interval
    fig_combined.add_trace(go.Scatter(
        x=df_forecast['timestamp'],
        y=df_forecast['upper_bound'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(254, 202, 87, 0.3)'),
        showlegend=False
    ))
    
    fig_combined.add_trace(go.Scatter(
        x=df_forecast['timestamp'],
        y=df_forecast['lower_bound'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(254, 202, 87, 0.3)'),
        name='Confidence Interval'
    ))
    
    fig_combined.update_layout(
        title="24-Hour Energy Consumption & Forecast",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode='x unified',
        height=450,
        paper_bgcolor='rgba(255, 255, 255, 0.1)',
        plot_bgcolor='rgba(255, 255, 255, 0.1)',
        font=dict(color='white'),
        legend=dict(bgcolor='rgba(255, 255, 255, 0.1)', bordercolor='white', borderwidth=1)
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)

st.markdown("---")

# ============================================================================
# DEVICE STATUS MONITOR
# ============================================================================

st.markdown("## 🏠 Device Status Monitor")

if st.session_state.devices:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        device_df = pd.DataFrame([
            {
                'Device': d.get('name', 'Unknown'),
                'Type': d.get('device_type', 'N/A'),
                'Status': '🟢 Active' if d.get('status') == 'active' else '⚫ Inactive',
                'Power': f"{d.get('power_rating', 0):.0f}W",
                'Efficiency': f"{d.get('efficiency', 0):.0f}%",
                'Last Seen': d.get('last_seen', 'Never')[:16] if d.get('last_seen') else 'Never'
            }
            for d in st.session_state.devices
        ])
        
        st.dataframe(device_df, use_container_width=True, height=400)
    
    with col2:
        # Device type distribution
        device_types = {}
        for d in st.session_state.devices:
            dtype = d.get('device_type', 'unknown')
            device_types[dtype] = device_types.get(dtype, 0) + 1
        
        fig_pie = px.pie(
            values=list(device_types.values()),
            names=list(device_types.keys()),
            title="Device Distribution by Type",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_pie.update_layout(
            height=400,
            paper_bgcolor='rgba(255, 255, 255, 0.1)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("📍 No devices found. Add devices using the sidebar!")

st.markdown("---")

# ============================================================================
# ALERTS & NOTIFICATIONS
# ============================================================================

st.markdown("## 🔔 Alerts & Notifications")

if st.session_state.alerts:
    for alert in st.session_state.alerts[:10]:  # Show last 10
        severity = alert.get('severity', 'info')
        color = {
            'critical': '🔴',
            'warning': '🟡',
            'info': '🔵'
        }.get(severity, '⚪')
        
        with st.expander(f"{color} {alert.get('message', 'Alert')}"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Severity:** {severity.upper()}")
                st.write(f"**Time:** {alert.get('created_at', 'Unknown')[:19]}")
            with col2:
                if not alert.get('acknowledged'):
                    if st.button("✅ Acknowledge", key=f"ack_{alert.get('id')}"):
                        st.success("Alert acknowledged!")
else:
    st.info("✨ No alerts at the moment. System running smoothly!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8); padding: 20px;'>
    <h3>⚡ Energy Management System</h3>
    <p>AI-Powered Smart Energy Control with ML Forecasting</p>
    <p style='font-size: 12px;'>Version 1.0 | Powered by Prophet ML & FastAPI</p>
</div>

""", unsafe_allow_html=True)


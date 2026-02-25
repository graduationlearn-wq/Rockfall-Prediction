# In app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from streamlit_geolocation import streamlit_geolocation
import requests
import time
import joblib
from data_simulators import simulate_geotech_data, simulate_environmental_data
from twilio.rest import Client

st.set_page_config(page_title="⛏️ Smart Rockfall Predictor", layout="wide", page_icon="🛰️")

PALETTE_RGB = np.array([[0, 0, 255], [255, 165, 0], [255, 0, 0]], dtype=np.uint8)
features = ['slope_angle', 'crack_density', 'displacement', 'strain', 'pore_pressure', 'rainfall', 'temperature', 'vibration']

@st.cache_resource
def load_model():
    try:
        return joblib.load('rockfall_model.joblib')
    except FileNotFoundError:
        return None
model = load_model()

if 'alert_sent' not in st.session_state:
    st.session_state.alert_sent = False

# Helper Functions (No Changes Here)
def generate_dem(lat, lon, size_km=1.0, resolution=50):
    lat_deg_per_km, lon_deg_per_km = 1 / 110.574, 1 / (111.320 * np.cos(np.radians(lat)))
    half_size_lat, half_size_lon = (size_km / 2) * lat_deg_per_km, (size_km / 2) * lon_deg_per_km
    lat_start, lat_end = lat - half_size_lat, lat + half_size_lat
    lon_start, lon_end = lon - half_size_lon, lon + half_size_lon
    lats, lons = np.linspace(lat_start, lat_end, resolution), np.linspace(lon_start, lon_end, resolution)
    points = [item for sublist in [[(lt, ln) for ln in lons] for lt in lats] for item in sublist]
    elevations = []
    for i in range(0, len(points), 50):
        chunk = points[i:i + 50]
        locations_str = "|".join([f"{lt},{ln}" for lt, ln in chunk])
        api_url = f"https://api.opentopodata.org/v1/aster30m?locations={locations_str}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data['status'] == 'OK':
                elevations.extend([result['elevation'] for result in data['results']])
            else: raise Exception(f"API Error: {data.get('error', 'Unknown error')}")
        except requests.exceptions.RequestException as e: raise Exception(f"Network error: {e}")
        time.sleep(0.1)
    if not elevations or len(elevations) != resolution * resolution: raise Exception("Failed to retrieve all elevation points.")
    return np.array(elevations).reshape((resolution, resolution)).astype(np.float32)

def load_dem(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError("Could not decode image.")
    if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)

def normalize_for_display(img):
    arr = img.astype(np.float32)
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn: return np.zeros_like(arr, dtype=np.uint8)
    return (np.clip((arr - mn) / (mx - mn), 0, 1) * 255).astype(np.uint8)

def colorize_classes(cls_map, palette_rgb=PALETTE_RGB):
    out = np.zeros((*cls_map.shape, 3), dtype=np.uint8)
    for val in (0, 1, 2): out[cls_map == val] = palette_rgb[val]
    return out

def make_overlay(gray_u8, cls_map, alpha_low=0.18, alpha_med=0.35, alpha_high=0.6):
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    out = base.copy()
    alphas = [alpha_low, alpha_med, alpha_high]
    for i in range(3):
        mask = (cls_map == i)
        color = (PALETTE_RGB[i] / 255.0).astype(np.float32)
        out[mask] = (1 - alphas[i]) * out[mask] + alphas[i] * color
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)

def compute_pipeline(img_float, medium_risk_thr, high_risk_thr):
    gray = img_float.copy()
    mn, mx = float(np.nanmin(gray)), float(np.nanmax(gray))
    elev01 = (gray - mn) / (mx - mn) if mx > mn else np.zeros_like(gray, dtype=np.float32)
    gy, gx = np.gradient(elev01)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    final_map = np.zeros_like(slope_deg, dtype=np.uint8)
    final_map[slope_deg >= medium_risk_thr] = 1
    final_map[slope_deg >= high_risk_thr] = 2
    areas = {"Low": np.mean(final_map == 0) * 100, "Medium": np.mean(final_map == 1) * 100, "High": np.mean(final_map == 2) * 100}
    dem_u8 = normalize_for_display(gray)
    images = {"dem": dem_u8, "final": colorize_classes(final_map), "overlay": make_overlay(dem_u8, final_map)}
    return {"slope_deg": slope_deg, "areas": areas, "images": images}

def send_sms_alert(probability, zone_name="North Pit"):
    account_sid = st.secrets.get("TWILIO_ACCOUNT_SID", "YOUR_SID_HERE")
    auth_token = st.secrets.get("TWILIO_AUTH_TOKEN", "YOUR_TOKEN_HERE")
    twilio_phone = st.secrets.get("TWILIO_PHONE", "+15017122661")
    recipient_phone = st.secrets.get("RECIPIENT_PHONE", "+1234567890")
    if "YOUR" in account_sid: return "Twilio credentials not set up."
    client = Client(account_sid, auth_token)
    message = f"CRITICAL ALERT: High rockfall probability ({probability:.0%}) in {zone_name}. ACTION: Evacuate personnel immediately."
    try:
        msg = client.messages.create(body=message, from_=twilio_phone, to=recipient_phone)
        return f"Alert SMS sent to {recipient_phone}!"
    except Exception as e:
        return f"Failed to send SMS: {e}"

def clear_dem_state():
    if 'dem_image' in st.session_state: del st.session_state.dem_image
    if 'alert_sent' in st.session_state: st.session_state.alert_sent = False

# Sidebar Controls
st.sidebar.header("Controls")
source_mode = st.sidebar.radio("DEM Source", ("Generate from My Location", "Upload DEM File"), on_change=clear_dem_state)
dem_img = None
if source_mode == "Upload DEM File":
    uploaded = st.sidebar.file_uploader("Upload DEM", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if uploaded:
        dem_img = load_dem(uploaded); st.session_state.dem_image = dem_img
else:
    st.sidebar.markdown("##### Generation Settings")
    dem_size_km = st.sidebar.slider("Area Size (km)", 0.5, 10.0, 2.0, step=0.5)
    dem_resolution = st.sidebar.select_slider("Resolution", options=[32, 50, 64, 100], value=50)
    location = streamlit_geolocation()
    if location and location.get('latitude'):
        st.sidebar.success(f"📍 Location Found")
        if st.sidebar.button("Generate DEM from My Location"):
            dem_img = generate_dem(location['latitude'], location['longitude'], dem_size_km, dem_resolution)
            st.session_state.dem_image = dem_img
    else:
        st.sidebar.info("Waiting for location...")

if 'dem_image' in st.session_state:
    dem_img = st.session_state.dem_image

st.sidebar.markdown("---")
st.sidebar.markdown("**Visual Risk Sensitivity**")
med_risk_thr = st.sidebar.slider("Medium Risk Threshold (°)", 0, 90, 15, 1)
high_risk_thr = st.sidebar.slider("High Risk Threshold (°)", med_risk_thr, 90, 30, 1)

# Main App Body
st.markdown("### ⛏️ Smart Rockfall Predictor")
if model is None: st.error("`rockfall_model.joblib` not found. Please run `train_model.py` first."); st.stop()
if dem_img is None: st.info("⬅️ Please select a DEM source in the sidebar to begin analysis."); st.stop()

# --- CORRECTED STRUCTURE ---
# Run the visual analysis once to get the static slope data
res = compute_pipeline(dem_img, med_risk_thr, high_risk_thr)
areas, images, slope_deg = res["areas"], res["images"], res["slope_deg"]

# Draw ALL the UI elements first
c1, c2, c3 = st.columns(3)
c1.metric("High-risk area (Visual)", f"{areas['High']:.1f}%")
c2.metric("Medium-risk area (Visual)", f"{areas['Medium']:.1f}%")
c3.metric("Low-risk area (Visual)", f"{areas['Low']:.1f}%")

st.markdown("---")
st.subheader("🧠 Live AI-Powered Risk Assessment")
is_simulated_anomaly = st.toggle("Simulate high-risk conditions for demo")

# A placeholder for the elements that will refresh in real-time
live_placeholder = st.empty()

# The tabs are now placed BEFORE the infinite loop
st.markdown("---")
tab1, tab2 = st.tabs(["Risk Maps", "Data Distribution"])
with tab1:
    st.subheader("Risk Maps")
    colA, colB = st.columns(2)
    colA.image(images["dem"], caption="Source DEM (Normalized)", use_column_width=True)
    colB.image(images["overlay"], caption="Risk Overlay", use_column_width=True)

with tab2:
    st.subheader("Slope Analysis")
    fig = go.Figure(go.Histogram(x=slope_deg.flatten(), nbinsx=40, marker_color="#6c8ebf"))
    fig.update_layout(title="Slope (°) Distribution", xaxis_title="Degrees", yaxis_title="Pixel Count")
    st.plotly_chart(fig, use_container_width=True)

# The real-time loop now ONLY updates the placeholder
while True:
    live_geotech = simulate_geotech_data(is_anomalous=is_simulated_anomaly)
    live_env = simulate_environmental_data(is_anomalous=is_simulated_anomaly)
    live_features_dict = {
        'slope_angle': slope_deg.mean(),
        'crack_density': np.random.uniform(0.1, 0.9) if is_simulated_anomaly else np.random.uniform(0.0, 0.2),
        'displacement': live_geotech['displacement_mm'].mean(),
        'strain': live_geotech['strain_ue'].mean(),
        'pore_pressure': live_geotech['pore_pressure_kpa'].mean(),
        'rainfall': live_env['rainfall_mm_hr'],
        'temperature': np.random.uniform(15, 30),
        'vibration': live_env['vibration_hz']
    }
    live_df = pd.DataFrame([live_features_dict])[features]
    risk_probability = model.predict(live_df)[0]

    with live_placeholder.container():
        st.write(f"**Calculated Rockfall Probability (AI Model):**")
        st.progress(float(risk_probability))
        st.metric("", f"{risk_probability:.1%}")

        ALERT_THRESHOLD = 0.50 
        if risk_probability > ALERT_THRESHOLD:
            if not st.session_state.alert_sent:
                alert_status = send_sms_alert(risk_probability)
                st.toast(f"🚨 High Risk Detected! {alert_status}")
                st.session_state.alert_sent = True
            st.error(f"ACTION REQUIRED: High risk of rockfall detected!")
        else:
            if st.session_state.alert_sent:
                st.toast("✅ Conditions are stable again. Alert system reset.")
            st.session_state.alert_sent = False
            st.success("Conditions appear stable.")

        with st.expander("Show Simulated Live Sensor Data Fed to AI"):
            st.json({k: round(v, 2) for k, v in live_features_dict.items()})
    
    time.sleep(1)
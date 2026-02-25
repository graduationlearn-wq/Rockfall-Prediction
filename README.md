# ⛏️ MineGuard AI: Predictive Rockfall Monitoring System

MineGuard AI is a proactive safety solution for open-pit mining. It moves beyond simple visual inspections by using Machine Learning to predict rockfall risks using terrain, weather, and sensor data.

### 🚀 Key Features
* **Global Terrain Analysis**: Fetches ASTER 30m DEM data for any coordinate.
* **AI Risk Core**: Uses XGBoost to forecast rockfall probability.
* **Live Dashboard**: Interactive Streamlit interface with risk overlays.
* **Emergency Alerts**: Automated SMS notifications via Twilio API when risk exceeds 75%.

### 🛠️ Installation & Setup
1. **Clone the repo**: `git clone https://github.com/graduationlearn-wq/Rockfall-Prediction.git`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run the app**: `streamlit run app.py`

### 📊 Tech Stack
* **Language**: Python
* **ML**: XGBoost, Scikit-Learn
* **Visualization**: Plotly, OpenCV
* **Backend**: Streamlit

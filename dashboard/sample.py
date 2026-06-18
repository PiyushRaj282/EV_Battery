import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="EV Battery RUL Predictor",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 AI-Powered EV Battery Health Monitoring")
st.markdown("### XGBoost-Based Remaining Useful Life Prediction")

st.divider()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "xgboost_rul.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("📂 Upload Battery Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if uploaded_file is None:
    st.info("Upload battery telemetry data to start analysis")
    st.stop()

df_master = pd.read_csv(uploaded_file)

df_master.columns = (
    df_master.columns
    .str.strip()
    .str.replace("\ufeff", "")
)

# --------------------------------------------------
# BATTERY SELECTION
# --------------------------------------------------

if "Battery_ID" not in df_master.columns:
    st.error("Battery_ID column missing")
    st.stop()

battery_list = df_master["Battery_ID"].unique()

selected_battery = st.selectbox(
    "Select Battery",
    battery_list
)

df = df_master[
    df_master["Battery_ID"] == selected_battery
].reset_index(drop=True)

st.success(f"Tracking Battery: {selected_battery}")

# --------------------------------------------------
# REQUIRED FEATURES
# --------------------------------------------------

required_features = [
    "SOH",
    "Capacity",
    "AvgVoltage",
    "AvgCurrent",
    "AvgTemp",
    "Re",
    "Rct"
]

missing_cols = [
    col for col in required_features
    if col not in df.columns
]

if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# --------------------------------------------------
# LATEST BATTERY STATE
# --------------------------------------------------

latest = df.iloc[-1]

X = np.array([
    latest["SOH"],
    latest["Capacity"],
    latest["AvgVoltage"],
    latest["AvgCurrent"],
    latest["AvgTemp"],
    latest["Re"],
    latest["Rct"]
]).reshape(1, -1)

predicted_rul = float(model.predict(X)[0])

# --------------------------------------------------
# BATTERY STATUS
# --------------------------------------------------

if predicted_rul < 20:
    status = "🔴 Critical"
elif predicted_rul < 50:
    status = "🟡 Warning"
else:
    status = "🟢 Healthy"

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "SOH",
        f"{latest['SOH']:.2f}%"
    )

with col2:
    st.metric(
        "Capacity",
        f"{latest['Capacity']:.3f}"
    )

with col3:
    st.metric(
        "Predicted RUL",
        f"{int(predicted_rul)} Cycles"
    )

with col4:
    st.metric(
        "Battery Status",
        status
    )

st.divider()

# --------------------------------------------------
# RUL GAUGE
# --------------------------------------------------

st.subheader("🎯 Remaining Useful Life")

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        title={"text": "Predicted RUL (Cycles)"},
        gauge={
            "axis": {"range": [0, 300]},
            "bar": {"color": "green"},
            "steps": [
                {"range": [0, 20], "color": "#ff4d4d"},
                {"range": [20, 50], "color": "#ffa64d"},
                {"range": [50, 300], "color": "#7CFC00"}
            ]
        }
    )
)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📈 Health Trends",
    "🔍 Diagnostics",
    "🤖 Model Insights"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    if "Cycle" in df.columns:

        fig1 = px.line(
            df,
            x="Cycle",
            y="SOH",
            title="SOH vs Cycle"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        fig2 = px.line(
            df,
            x="Cycle",
            y="Capacity",
            title="Capacity vs Cycle"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    colA, colB = st.columns(2)

    with colA:

        if "AvgTemp" in df.columns:

            fig_temp = px.line(
                df,
                x="Cycle",
                y="AvgTemp",
                title="Temperature Profile"
            )

            st.plotly_chart(
                fig_temp,
                use_container_width=True
            )

    with colB:

        if "Re" in df.columns:

            fig_re = px.line(
                df,
                x="Cycle",
                y="Re",
                title="Electrolyte Resistance (Re)"
            )

            st.plotly_chart(
                fig_re,
                use_container_width=True
            )

    colC, colD = st.columns(2)

    with colC:

        if "Rct" in df.columns:

            fig_rct = px.line(
                df,
                x="Cycle",
                y="Rct",
                title="Charge Transfer Resistance (Rct)"
            )

            st.plotly_chart(
                fig_rct,
                use_container_width=True
            )

    with colD:

        if "AvgVoltage" in df.columns:

            fig_voltage = px.line(
                df,
                x="Cycle",
                y="AvgVoltage",
                title="Average Voltage"
            )

            st.plotly_chart(
                fig_voltage,
                use_container_width=True
            )

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader("Latest Feature Values")

    feature_df = pd.DataFrame({
        "Feature": required_features,
        "Value": [
            latest["SOH"],
            latest["Capacity"],
            latest["AvgVoltage"],
            latest["AvgCurrent"],
            latest["AvgTemp"],
            latest["Re"],
            latest["Rct"]
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": required_features,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance"
        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True
        )

st.divider()

st.caption(
    "AI-Powered EV Battery Health Monitoring & RUL Prediction Dashboard"
)
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="EV Battery Health Monitoring",
    layout="wide"
)

# --------------------
# LOAD
# --------------------

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgboost_rul.pkl"

DATA_PATH = BASE_DIR / "data" / "final_df.csv"

print(MODEL_PATH)
print(DATA_PATH)

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

# --------------------
# SIDEBAR
# --------------------

battery = st.sidebar.selectbox(
    "Select Battery",
    df["Battery_ID"].unique()
)

battery_df = (
    df[df["Battery_ID"] == battery]
    .sort_values("Cycle")
)

# latest record

latest = battery_df.iloc[-1]

# --------------------
# PREDICTION
# --------------------

FEATURES = [
    "SOH",
    "Capacity",
    "AvgVoltage",
    "AvgCurrent",
    "AvgTemp",
    "Re",
    "Rct"
]

pred_rul = model.predict(
    latest[FEATURES]
    .values
    .reshape(1,-1)
)[0]

# --------------------
# HEADER
# --------------------

st.title(
    "🔋 EV Battery Health Monitoring & RUL Prediction"
)

# --------------------
# METRICS
# --------------------

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "SOH",
    f"{latest['SOH']:.2f}%"
)

c2.metric(
    "Capacity",
    f"{latest['Capacity']:.3f} Ah"
)

c3.metric(
    "Predicted RUL",
    f"{pred_rul:.0f} Cycles"
)

# status

if latest["SOH"] > 90:
    status = "Healthy"

elif latest["SOH"] > 80:
    status = "Warning"

else:
    status = "Critical"

c4.metric(
    "Status",
    status
)

# --------------------
# SOH CURVE
# --------------------

st.subheader(
    "Battery Health Trend"
)

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    battery_df["Cycle"],
    battery_df["SOH"]
)

ax.set_xlabel("Cycle")
ax.set_ylabel("SOH")

ax.grid()

st.pyplot(fig)

# --------------------
# FEATURE IMPORTANCE
# --------------------

st.subheader(
    "Feature Importance"
)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance":
        model.feature_importances_
})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=True
    )
)

fig2, ax2 = plt.subplots(
    figsize=(8,5)
)

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

st.pyplot(fig2)

# --------------------
# CURRENT VALUES
# --------------------

st.subheader(
    "Current Sensor Values"
)

st.dataframe(
    pd.DataFrame({
        "Parameter":[
            "Voltage",
            "Current",
            "Temperature",
            "Re",
            "Rct"
        ],
        "Value":[
            latest["AvgVoltage"],
            latest["AvgCurrent"],
            latest["AvgTemp"],
            latest["Re"],
            latest["Rct"]
        ]
    })
)
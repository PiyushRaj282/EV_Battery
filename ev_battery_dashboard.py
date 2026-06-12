import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="EV Battery RUL Predictor", page_icon="🔋", layout="wide")

st.title("🔋 EV Battery Remaining Useful Life (RUL) Diagnostic Dashboard")
st.markdown("**Physics-Informed Neural Network (PINN) Multi-Asset Forecasting**")
st.write("---")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Physics Parameters")
alpha = st.sidebar.slider("Degradation Coefficient (α)", 
                          min_value=0.0001, max_value=0.0050, value=0.0012, step=0.0001)
eol_threshold = st.sidebar.slider("End-of-Life Threshold (%)", 
                                  min_value=60.0, max_value=90.0, value=80.0, step=1.0)

# --- File Uploader ---
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Battery Telemetry (CSV)", type=["csv"])

# --- Core Data Loading Logic ---
df = None

if uploaded_file is not None:
    # Read the file
    master_df = pd.read_csv(uploaded_file)
    
    # 🚨 BULLETPROOF FIX 🚨
    # Strip hidden characters (like BOM from Excel) and whitespace from headers
    master_df.columns = master_df.columns.str.strip().str.replace('\ufeff', '')
    
    # Verify the generated data has the correct column
    if 'Battery_ID' in master_df.columns:
        # Move the dropdown to the MAIN screen
        unique_batteries = master_df['Battery_ID'].unique()
        selected_bat = st.selectbox(
            "⬇️ SELECT A SPECIFIC BATTERY PROFILE TO ANALYZE ⬇️", 
            options=unique_batteries
        )
        
        # Filter the dataset
        df = master_df[master_df['Battery_ID'] == selected_bat].reset_index(drop=True)
        st.success(f"Currently tracking: {selected_bat}")
    else:
        st.error(f"⚠️ Could not find 'Battery_ID'. Found columns: {list(master_df.columns)}")
else:
    st.info("💡 Please upload your telemetry CSV file to begin.")

st.write("---")

# --- Main Analytics and Visualizations ---
# Only run the graphs if data was successfully loaded
if df is not None:
    if 'Cycle' not in df.columns or 'Capacity' not in df.columns:
        st.error(f"⚠️ Active data must contain 'Cycle' and 'Capacity' columns. Found: {list(df.columns)}")
    else:
        # 1. Pipeline Calculus
        initial_capacity = df['Capacity'].iloc[0]
        current_capacity = df['Capacity'].iloc[-1]
        current_soh = (current_capacity / initial_capacity) * 100
        current_cycle = int(df['Cycle'].iloc[-1])
        target_capacity = initial_capacity * (eol_threshold / 100.0)
        
        if current_capacity > target_capacity:
            predicted_cycles_left = int(-np.log(target_capacity / current_capacity) / alpha)
        else:
            predicted_cycles_left = 0
            
        total_predicted_life = current_cycle + predicted_cycles_left

        # 2. Executive KPI Summary Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            soh_delta = current_soh - (df['Capacity'].iloc[-2] / initial_capacity * 100) if len(df) > 1 else 0
            st.metric(label="Calculated State of Health (SOH)", 
                      value=f"{current_soh:.2f}%", 
                      delta=f"{soh_delta:.3f}% vs prior interval",
                      delta_color="inverse")
                      
        with col2:
            st.metric(label="Predicted Remaining Useful Life (RUL)", value=f"{predicted_cycles_left} Cycles left")
                      
        with col3:
            if current_soh <= eol_threshold:
                status = "Critical / EOL Breach"
            elif current_soh <= eol_threshold + 5:
                status = "Service Advised"
            else:
                status = "Healthy Asset"
            st.metric(label="System Status Verdict", value=status)

        st.write("---")

        # 3. Main Projection Chart (Historical + Future Prediction)
        st.subheader("📈 Capacity Degradation & PINN Forecast")
        
        future_cycles = np.arange(current_cycle + 1, total_predicted_life + 50)
        future_capacity = current_capacity * np.exp(-alpha * (future_cycles - current_cycle))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Capacity'], mode='lines', name='Observed Line Telemetry', line=dict(color='#1f77b4', width=2.5)))
        fig.add_trace(go.Scatter(x=future_cycles, y=future_capacity, mode='lines', name='Physics Projection Curve', line=dict(color='#ff7f0e', width=2.5, dash='dot')))
        fig.add_hline(y=target_capacity, line_dash="dash", line_color="red", annotation_text=f"End of Life Limit ({eol_threshold}%)")
        
        fig.update_layout(xaxis_title="Total Runtime Cycles", yaxis_title="Capacity Magnitude", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 4. Secondary Telemetry Diagnostics
        st.subheader("🔍 Secondary Channel Telemetry Diagnostics")
        col_diag1, col_diag2 = st.columns(2)
        
        with col_diag1:
            if 'Temperature_C' in df.columns:
                fig_temp = px.line(df, x='Cycle', y='Temperature_C', title="Operating Temperature Profile (°C)", color_discrete_sequence=['#d62728'])
                st.plotly_chart(fig_temp, use_container_width=True)
                
        with col_diag2:
            if 'Internal_Resistance_Ohm' in df.columns:
                fig_ir = px.line(df, x='Cycle', y='Internal_Resistance_Ohm', title="Internal Resistance Curve (Ohms)", color_discrete_sequence=['#9467bd'])
                st.plotly_chart(fig_ir, use_container_width=True)

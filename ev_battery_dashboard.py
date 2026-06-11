import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="EV Battery RUL Predictor", page_icon="🔋", layout="wide")

st.title("🔋 EV Battery Remaining Useful Life (RUL) Diagnostic Dashboard")
st.markdown("**Physics-Informed Neural Network (PINN) Real-Time Forecasting**")
st.write("---")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Data & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Battery Telemetry (CSV)", type=["csv"])

st.sidebar.subheader("Physics Parameters")
# The alpha slider controls how aggressively the physical degradation curve falls
alpha = st.sidebar.slider("Degradation Coefficient (α)", 
                          min_value=0.0001, max_value=0.0050, value=0.0012, step=0.0001, 
                          help="Simulates the rate of capacity fade based on thermodynamic constraints.")
eol_threshold = st.sidebar.slider("End-of-Life Threshold (%)", 
                                  min_value=60.0, max_value=90.0, value=80.0, step=1.0,
                                  help="The percentage of original capacity where the battery is considered dead.")

# --- Demo Data Generator ---
# This ensures the dashboard works perfectly out of the box even before you upload your own dataset.
@st.cache_data
def generate_dummy_data():
    cycles = np.arange(1, 201)
    # Starting capacity at 100%, degrading following a rough exponential curve with added sensor noise
    base_capacity = 100 * np.exp(-0.0008 * cycles) 
    noise = np.random.normal(0, 0.5, len(cycles))
    capacity = base_capacity + noise
    
    # Adding synthetic telemetry metrics
    temperature = 25 + (cycles * 0.05) + np.random.normal(0, 2, len(cycles))
    internal_resistance = 0.02 + (cycles * 0.0001) + np.random.normal(0, 0.001, len(cycles))
    
    return pd.DataFrame({
        'Cycle': cycles,
        'Capacity': capacity,
        'Temperature_C': temperature,
        'Internal_Resistance_Ohm': internal_resistance
    })

# --- Data Loading Logic ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = generate_dummy_data()
else:
    st.sidebar.info("💡 No file uploaded. Using simulated baseline telemetry data for demonstration.")
    df = generate_dummy_data()

# --- Main Dashboard Logic ---
# Validation check for required columns
if 'Cycle' not in df.columns or 'Capacity' not in df.columns:
    st.error("⚠️ Uploaded CSV must contain at least 'Cycle' and 'Capacity' columns.")
else:
    # 1. Core Calculations
    initial_capacity = df['Capacity'].iloc[0]
    current_capacity = df['Capacity'].iloc[-1]
    current_soh = (current_capacity / initial_capacity) * 100
    current_cycle = int(df['Cycle'].iloc[-1])
    target_capacity = initial_capacity * (eol_threshold / 100.0)
    
    # 2. Physics-Informed Prediction Logic (Math)
    # Equation: Q(N) = Q_current * exp(-alpha * delta_N)
    # Solving for delta_N (Remaining Cycles) to reach Target Capacity:
    if current_capacity > target_capacity:
        predicted_cycles_left = int(-np.log(target_capacity / current_capacity) / alpha)
    else:
        predicted_cycles_left = 0
        
    total_predicted_life = current_cycle + predicted_cycles_left

    # 3. Top Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Calculate degradation rate over the last cycle
        soh_delta = current_soh - (df['Capacity'].iloc[-2] / initial_capacity * 100) if len(df) > 1 else 0
        st.metric(label="Current State of Health (SOH)", 
                  value=f"{current_soh:.2f}%", 
                  delta=f"{soh_delta:.3f}% vs last cycle",
                  delta_color="inverse") # Inverse because dropping capacity is bad
                  
    with col2:
        st.metric(label="Predicted Remaining Useful Life (RUL)", 
                  value=f"{predicted_cycles_left} Cycles left")
                  
    with col3:
        if current_soh <= eol_threshold:
            status, color = "Critical / EOL Reached", "red"
        elif current_soh <= eol_threshold + 5:
            status, color = "Service Advised", "orange"
        else:
            status, color = "Healthy", "green"
            
        st.metric(label="Battery Condition Alert", value=status)

    st.write("---")

    # 4. Main Projection Chart (Historical + Future Prediction)
    st.subheader("📈 Capacity Degradation & PINN Forecast")
    
    # Generate the curve for the AI's predicted future
    future_cycles = np.arange(current_cycle + 1, total_predicted_life + 50)
    future_capacity = current_capacity * np.exp(-alpha * (future_cycles - current_cycle))
    
    fig = go.Figure()
    
    # Plot Actual Historical Data
    fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Capacity'], 
                             mode='lines', name='Actual Sensor Data',
                             line=dict(color='#1f77b4', width=2.5)))
    
    # Plot Predicted Future Data
    fig.add_trace(go.Scatter(x=future_cycles, y=future_capacity, 
                             mode='lines', name='Physics-Informed Prediction',
                             line=dict(color='#ff7f0e', width=2.5, dash='dot')))
    
    # Add End-of-Life (EOL) Threshold Horizontal Line
    fig.add_hline(y=target_capacity, line_dash="dash", line_color="red", 
                  annotation_text=f"End of Life Threshold ({eol_threshold}%)", 
                  annotation_position="bottom left",
                  annotation_font=dict(color="red"))
    
    fig.update_layout(xaxis_title="Total Charge Cycles", 
                      yaxis_title="Battery Capacity (Ah / %)",
                      hovermode="x unified",
                      margin=dict(l=20, r=20, t=30, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                      
    st.plotly_chart(fig, use_container_width=True)

    # 5. Diagnostic Telemetry Charts
    st.subheader("🔍 Secondary Telemetry Diagnostics")
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        if 'Temperature_C' in df.columns:
            fig_temp = px.line(df, x='Cycle', y='Temperature_C', 
                               title="Operating Temperature Trend (°C)",
                               color_discrete_sequence=['#d62728'])
            # Add a trendline to see if battery is getting hotter as it ages
            fig_temp.add_trace(go.Scatter(x=df['Cycle'], y=df['Temperature_C'].rolling(window=10).mean(), 
                                          mode='lines', name='Moving Average',
                                          line=dict(color='black', width=1, dash='dash')))
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Temperature data not available in uploaded dataset.")
            
    with col_diag2:
        if 'Internal_Resistance_Ohm' in df.columns:
            fig_ir = px.line(df, x='Cycle', y='Internal_Resistance_Ohm', 
                             title="Internal Resistance Growth (Ohms)",
                             color_discrete_sequence=['#9467bd'])
            st.plotly_chart(fig_ir, use_container_width=True)
        else:
            st.info("Internal Resistance data not available in uploaded dataset.")

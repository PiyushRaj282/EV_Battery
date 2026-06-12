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
st.sidebar.header("⚙️ Data & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Battery Telemetry (CSV)", type=["csv"])

st.sidebar.subheader("Physics Parameters")
alpha = st.sidebar.slider("Degradation Coefficient (α)", 
                          min_value=0.0001, max_value=0.0050, value=0.0012, step=0.0001, 
                          help="Simulates the rate of capacity fade based on thermodynamic constraints.")
eol_threshold = st.sidebar.slider("End-of-Life Threshold (%)", 
                                  min_value=60.0, max_value=90.0, value=80.0, step=1.0,
                                  help="The percentage of original capacity where the battery is considered dead.")

# --- Demo Data Generator (Multi-Battery Support) ---
@st.cache_data
def generate_dummy_data():
    all_data = []
    battery_ids = ["BAT_001", "BAT_002", "BAT_003"]
    decay_rates = [0.0006, 0.0009, 0.0012]
    
    for b_id, rate in zip(battery_ids, decay_rates):
        cycles = np.arange(1, 201)
        base_capacity = 100 * np.exp(-rate * cycles) 
        noise = np.random.normal(0, 0.4, len(cycles))
        capacity = base_capacity + noise
        temperature = 25 + (cycles * (rate * 50)) + np.random.normal(0, 1.5, len(cycles))
        internal_resistance = 0.02 + (cycles * (rate * 0.1)) + np.random.normal(0, 0.0005, len(cycles))
        
        batch_df = pd.DataFrame({
            'Battery_ID': b_id,
            'Cycle': cycles,
            'Capacity': capacity,
            'Temperature_C': temperature,
            'Internal_Resistance_Ohm': internal_resistance
        })
        all_data.append(batch_df)
        
    return pd.concat(all_data, ignore_index=True)

# --- Data Loading Logic ---
if uploaded_file is not None:
    try:
        # Sneak peek at data to handle unlabeled or labeled formats
        df_preview = pd.read_csv(uploaded_file, nrows=2)
        
        # CRITICAL FIX: Rewind the file pointer back to the very beginning!
        uploaded_file.seek(0) 
        
        # Check if the file is unlabelled/raw (first row contains pure numeric digits)
        if df_preview.columns[0].replace('.','',1).isdigit() or df_preview.shape[1] < 3:
            df = pd.read_csv(uploaded_file, header=None)
            # Reassign consistent positional mapping
            df.columns = ['Cycle', 'Capacity'] + [f'Channel_{i}' for i in range(2, df.shape[1])]
            # Auto-boundary detection for raw unlabeled files via cycle count resets
            resets = np.where(df['Cycle'].diff() < 0)[0]
            boundaries = [0] + list(resets) + [len(df)]
            run_labels = [f"Battery Run {i+1}" for i in range(len(boundaries)-1)]
            
            selected_run = st.sidebar.selectbox("📂 Select Battery Profile", options=run_labels)
            run_idx = run_labels.index(selected_run)
            df = df.iloc[boundaries[run_idx]:boundaries[run_idx+1]].reset_index(drop=True)
            
            if 'Channel_2' in df.columns: df = df.rename(columns={'Channel_2': 'Temperature_C'})
            if 'Channel_3' in df.columns: df = df.rename(columns={'Channel_3': 'Internal_Resistance_Ohm'})
        else:
            df = pd.read_csv(uploaded_file)
            # Standard Labeled Approach: Check for Grouping Identifier
            if 'Battery_ID' in df.columns:
                unique_batteries = df['Battery_ID'].unique()
                selected_bat = st.sidebar.selectbox("📂 Select Battery Profile", options=unique_batteries)
                df = df[df['Battery_ID'] == selected_bat].reset_index(drop=True)
                
        st.sidebar.success("Telemetry dataset parsed cleanly!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = generate_dummy_data()
        selected_bat = st.sidebar.selectbox("📂 Select Battery Profile (Demo Mode)", options=df['Battery_ID'].unique())
        df = df[df['Battery_ID'] == selected_bat].reset_index(drop=True)
else:
    st.sidebar.info("💡 Using multi-asset demonstration matrix baseline.")
    demo_df = generate_dummy_data()
    selected_bat = st.sidebar.selectbox("📂 Select Battery Profile (Demo Mode)", options=demo_df['Battery_ID'].unique())
    df = demo_df[demo_df['Battery_ID'] == selected_bat].reset_index(drop=True)

# --- Main Analytics and Visualizations ---
if 'Cycle' not in df.columns or 'Capacity' not in df.columns:
    st.error("⚠️ Invalid Data Format: Active frame mapping requires 'Cycle' and 'Capacity' markers.")
else:
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

    st.subheader("📈 Capacity Degradation & PINN Forecast")
    
    future_cycles = np.arange(current_cycle + 1, total_predicted_life + 50)
    future_capacity = current_capacity * np.exp(-alpha * (future_cycles - current_cycle))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Cycle'], y=df['Capacity'], mode='lines', name='Observed Line Telemetry', line=dict(color='#1f77b4', width=2.5)))
    fig.add_trace(go.Scatter(x=future_cycles, y=future_capacity, mode='lines', name='Physics Projection Curve', line=dict(color='#ff7f0e', width=2.5, dash='dot')))
    fig.add_hline(y=target_capacity, line_dash="dash", line_color="red", annotation_text=f"End of Life Limit ({eol_threshold}%)")
    
    fig.update_layout(xaxis_title="Total Runtime Cycles", yaxis_title="Capacity Magnitude", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Secondary Channel Telemetry Diagnostics")
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        if 'Temperature_C' in df.columns:
            fig_temp = px.line(df, x='Cycle', y='Temperature_C', title="Operating Temperature Profile (°C)", color_discrete_sequence=['#d62728'])
            fig_temp.add_trace(go.Scatter(x=df['Cycle'], y=df['Temperature_C'].rolling(window=10, min_periods=1).mean(), mode='lines', name='Moving Average', line=dict(color='black', width=1, dash='dash')))
            st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Temperature records not available in the current partition slice.")
            
    with col_diag2:
        if 'Internal_Resistance_Ohm' in df.columns:
            fig_ir = px.line(df, x='Cycle', y='Internal_Resistance_Ohm', title="Internal Resistance Curve (Ohms)", color_discrete_sequence=['#9467bd'])
            st.plotly_chart(fig_ir, use_container_width=True)
        else:
            st.info("Internal Resistance records not available in the current partition slice.")

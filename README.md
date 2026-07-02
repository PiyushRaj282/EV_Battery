# AI-Powered EV Battery Health Monitoring and Remaining Useful Life (RUL) Prediction System

## Overview

This project presents an AI-based Battery Health Monitoring System that predicts the **Remaining Useful Life (RUL)** of Electric Vehicle (EV) batteries using Machine Learning. The system analyzes battery degradation patterns from the NASA Battery Dataset and estimates the battery's remaining operational life based on its current health condition.

The project includes data preprocessing, feature engineering, model training using XGBoost, and an interactive Streamlit dashboard for battery health visualization and RUL prediction.

---

## Objectives

- Monitor battery health using key battery parameters.
- Estimate the State of Health (SOH) of lithium-ion batteries.
- Predict the Remaining Useful Life (RUL) using Machine Learning.
- Visualize battery degradation through an interactive dashboard.
- Support predictive maintenance and battery replacement decisions.

---

## Dataset

The project uses the **NASA Prognostics Center of Excellence (PCoE) Battery Dataset**.

The dataset contains battery aging information collected through repeated charging and discharging cycles, including:

- Battery Capacity
- Voltage
- Current
- Temperature
- Charge/Discharge Cycles
- Electrochemical Impedance Spectroscopy (EIS)
- Electrolyte Resistance (Re)
- Charge Transfer Resistance (Rct)

Batteries used in this project:

- B0005
- B0006
- B0007
- B0018

---

## Features Used

The following features were used to train the model:

- State of Health (SOH)
- Capacity
- Average Voltage
- Average Current
- Average Temperature
- Electrolyte Resistance (Re)
- Charge Transfer Resistance (Rct)

Target Variable:

- Remaining Useful Life (RUL)

---

## Machine Learning Model

The primary model used in this project is **XGBoost Regressor**.

Reasons for choosing XGBoost:

- High prediction accuracy on tabular data
- Fast training and prediction
- Handles non-linear relationships effectively
- Provides feature importance for model interpretation

An LSTM model was also implemented for comparison, but XGBoost demonstrated better performance on the available dataset.

---

## Project Workflow

```
NASA Battery Dataset
        │
        ▼
Data Extraction
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
SOH & RUL Calculation
        │
        ▼
XGBoost Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Streamlit Dashboard
```

---

## Dashboard Features

The dashboard provides:

- Battery selection
- Current battery health information
- State of Health (SOH)
- Battery Capacity
- Predicted Remaining Useful Life (RUL)
- Battery status (Healthy, Warning, Critical)
- SOH degradation graph
- Feature importance visualization
- Current battery sensor values

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- XGBoost
- TensorFlow (LSTM Comparison)
- Matplotlib
- Plotly
- Streamlit
- Joblib

---

## Project Structure

```
EV_Battery_RUL_Project/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final_df.csv
│
├── models/
│   └── xgboost_rul.pkl
│
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   └── 02_Model_Training.ipynb
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/EV_Battery_RUL_Project.git
cd EV_Battery_RUL_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the Model

Open Jupyter Notebook and execute the notebooks in sequence:

```
01_Dataset_Exploration.ipynb

↓

02_Model_Training.ipynb
```

This generates:

- `final_df.csv`
- `xgboost_rul.pkl`

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Results

The XGBoost model was evaluated using a battery-wise train-test split to assess its performance on an unseen battery.

Evaluation Metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

The dashboard provides battery health monitoring along with Remaining Useful Life prediction based on these trained models.

---

## Future Enhancements

- Live battery data streaming
- IoT sensor integration
- Fleet-level battery monitoring
- Prediction uncertainty estimation
- Cloud deployment
- API integration

---


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Kaggle imports
import kagglehub
from kagglehub import KaggleDatasetAdapter

st.set_page_config(page_title="AQI Forecasting App", layout="wide")

st.title("🌫️ Air Quality Index (AQI) Forecasting")
st.write("Predict tomorrow's AQI for a selected city using ML models.")

# -------------------- Data Loading --------------------
@st.cache_data
def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "ankushpanday1/air-quality-data-in-india-2015-2024",
        "city_day.csv"
    )

    # 🔥 IMPORTANT: Convert to proper pandas dataframe (fix Arrow issues)
    df = pd.DataFrame(df)

    # Format date
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df['Date'] = pd.to_datetime(df[date_col])
    df = df.sort_values('Date')

    return df

with st.spinner('Fetching latest air quality data from Kaggle...'):
    df = load_data()

# -------------------- City Selector --------------------
st.sidebar.header("⚙️ App Settings")
cities = sorted(df['City'].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", cities, index=0)

st.subheader(f"📍 City: {selected_city}")

df = df[df['City'] == selected_city].copy()

# -------------------- Data Cleaning --------------------
st.subheader("📌 Data Preprocessing")
st.write("Handling missing values using linear interpolation.")

# 🔥 FIX: Interpolate only numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].interpolate(method='linear', limit_direction='both')

df = df.dropna()

# -------------------- Feature Engineering --------------------
df['AQI_Tomorrow'] = df['AQI'].shift(-1)
df['Month'] = df['Date'].dt.month
df = df.dropna()

features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3', 'Month']
target = 'AQI_Tomorrow'

# -------------------- Train-Test Split --------------------
split_date = pd.to_datetime('2023-01-01')
train = df[df['Date'] < split_date]
test = df[df['Date'] >= split_date]

# 🔥 Safety check (avoid crash if city has low data)
if train.empty or test.empty:
    st.error("Not enough data for this city to train/test. Please select another city.")
    st.stop()

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

# -------------------- Model Training --------------------
@st.cache_resource
def train_models(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)

    return rf, xgb

rf_model, xgb_model = train_models(X_train, y_train)

# -------------------- Evaluation --------------------
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return preds, mae, r2

rf_preds, rf_mae, rf_r2 = evaluate_model(rf_model, X_test, y_test)
xgb_preds, xgb_mae, xgb_r2 = evaluate_model(xgb_model, X_test, y_test)

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.metric("Random Forest MAE", f"{rf_mae:.2f}")
    st.metric("Random Forest R²", f"{rf_r2:.4f}")
with col2:
    st.metric("XGBoost MAE", f"{xgb_mae:.2f}")
    st.metric("XGBoost R²", f"{xgb_r2:.4f}")

# -------------------- Visualization --------------------
st.subheader("📈 Actual vs Predicted AQI")

plot_df = pd.DataFrame({
    'Date': test['Date'],
    'Actual AQI': y_test,
    'Random Forest': rf_preds,
    'XGBoost': xgb_preds
}).set_index('Date')

st.line_chart(plot_df)

# -------------------- AQI Category --------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# -------------------- Prediction --------------------
st.subheader("🔮 Predict Tomorrow's AQI")

with st.form("prediction_form"):
    pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
    pm10 = st.number_input("PM10", min_value=0.0, value=100.0)
    no2 = st.number_input("NO2", min_value=0.0, value=30.0)
    co = st.number_input("CO", min_value=0.0, value=1.0)
    so2 = st.number_input("SO2", min_value=0.0, value=10.0)
    o3 = st.number_input("O3", min_value=0.0, value=20.0)
    month = st.selectbox("Month", list(range(1, 13)))

    submitted = st.form_submit_button("Predict AQI")

if submitted:
    input_data = pd.DataFrame([[pm25, pm10, no2, co, so2, o3, month]], columns=features)

    rf_pred = rf_model.predict(input_data)[0]
    xgb_pred = xgb_model.predict(input_data)[0]

    st.success("Prediction Completed!")

    col1, col2 = st.columns(2)
    with col1:
        st.write("🌲 **Random Forest**")
        st.metric("AQI", f"{rf_pred:.2f}")
        st.write(f"Category: **{aqi_category(rf_pred)}**")
    with col2:
        st.write("⚡ **XGBoost**")
        st.metric("AQI", f"{xgb_pred:.2f}")
        st.write(f"Category: **{aqi_category(xgb_pred)}**")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed as an ML-based AQI Forecasting Web App using Streamlit")
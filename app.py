import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="Coca-Cola Stock Predictor", layout="centered")

# Load model and data
@st.cache_data
def load_data():
    model = joblib.load('rf_model.pkl')
    data = pd.read_csv('Cleaned_Coca_Cola_stock_history.csv')
    return model, data

model, data = load_data()

# Prediction logic
def predict_next(data, model):
    latest_features = data.iloc[-1][[
        'Open', 'High', 'Low', 'Volume',
        'Daily % Change', 'Volatility',
        'MA_5', 'MA_20', 'MA_50'
    ]].fillna(method='bfill').values.reshape(1, -1)
    
    return model.predict(latest_features)[0]

# App UI
st.title("Coca-Cola Stock Prediction")
st.markdown("This app uses a trained Random Forest model to predict the **next closing price** of Coca-Cola stock based on the most recent data.")

st.subheader("Latest Available Data")
st.dataframe(data.tail(1).style.format(precision=2), use_container_width=True)

if st.button("Predict Next Close Price"):
    prediction = predict_next(data, model)
    st.success(f"Predicted Close Price: **${prediction:.2f}**")

# Optional: Add sidebar for expansion
with st.sidebar:
    st.header("About")
    st.markdown("""
    - Model: Random Forest Regressor  
    - Data Source: Coca-Cola historical stock data  
    - Features used: Open, High, Low, Volume, % Change, Volatility, MA_5/20/50  
    """)

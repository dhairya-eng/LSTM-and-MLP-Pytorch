import streamlit as st
import torch
import numpy as np
from model import LSTMStockModel

st.title("Stock Price Prediction (LSTM)")

# Load model
model = LSTMStockModel()
model.load_state_dict(torch.load("stock_lstm.pth", map_location="cpu"))
model.eval()

# Input form
st.write("Enter 10 days of stock data (Open, High, Low, Close, Volume) row by row")
data = st.text_area("Example:\n100,101,99,100.5,500000\n...", height=200)

if st.button("Predict"):
    try:
        rows = [list(map(float, row.split(','))) for row in data.strip().split('\n')]
        arr = np.array(rows).reshape(1, 10, 5)
        x = torch.tensor(arr, dtype=torch.float32)
        pred = model(x).item()
        st.success(f"Predicted Next Day Close: {pred:.2f}")
    except:
        st.error("Invalid input format. Please enter numeric values only.")

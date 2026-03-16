import streamlit as st
from app import get_predictions

st.title("Stock Predictions App")

tickers = st.text_input("Enter tickers (comma separated)", "AAPL,MSFT,NVDA,TSLA,AMZN")
period = st.selectbox("Select period", ["1mo", "3mo", "6mo", "1y"])

if st.button("Run Predictions"):
    results = get_predictions(tickers=tickers, period=period)
    st.text(results)

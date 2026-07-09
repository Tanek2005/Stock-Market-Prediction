import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title=“Stock Market Prediction”, layout=“wide”)

st.title(“Stock Market Prediction”)

stocks = {
“Apple”: “AAPL”,
“Tesla”: “TSLA”,
“Amazon”: “AMZN”,
“Google”: “GOOGL”,
“Microsoft”: “MSFT”,
“NVIDIA”: “NVDA”
}

option = st.radio(“Select Input Method”, [“Top Stocks”, “Enter Ticker”])

if option == “Top Stocks”:
selected_stock = st.selectbox(“Choose a stock”, list(stocks.keys()))
ticker = stocks[selected_stock]
else:
ticker = st.text_input(“Enter Stock Ticker (e.g. META, NFLX, AMD)”).upper()

if not ticker:
st.stop()

data = yf.download(ticker, start=“2020-01-01”, end=str(datetime.date.today()))

if data.empty:
st.error(“Invalid ticker or no data found.”)
st.stop()

st.subheader(f”{ticker} Price Chart”)
st.line_chart(data[“Close”])

data[“Prediction”] = data[“Close”].shift(-1)
data.dropna(inplace=True)

X = data[[“Close”]].values
y = data[“Prediction”].values

model = LinearRegression()
model.fit(X, y)

last_price = float(data[“Close”].iloc[-1])

X_input = np.array(last_price).reshape(1, -1)

prediction = model.predict(X_input)
predicted_price = float(prediction[0])

st.subheader(“Predicted Next Day Price”)
st.success(f”${predicted_price:.2f}”)

if predicted_price > last_price:
st.write(“Bullish Signal”)
else:
st.write(“Bearish Signal”)

today = datetime.date.today()
next_day = today + datetime.timedelta(days=1)

while next_day.weekday() >= 5:
next_day += datetime.timedelta(days=1)

st.info(f”Next Trading Day: {next_day}”)
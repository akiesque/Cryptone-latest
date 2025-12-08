import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

# Paths (adjust if you rename)
DATA_PATH = Path("dataset/price/btc_prices_2020_present.csv")
LSTM_PATH = Path("models/btc_hybrid_lstm.keras")
META_PATH = Path("models/btc_hybrid_meta.pkl")  # still named btc_* in current script

# Load meta
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

arima_order = tuple(meta["arima_order"])
scaler = meta["residual_scaler"]
seq_len = meta["seq_len"]

# Load data
df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").dropna(subset=["price"])
df = df.set_index("timestamp")
daily = df["price"].resample("D").last().dropna()

log_series = np.log(daily)
train_log = log_series

# Fit ARIMA with stored order
arima_model = ARIMA(train_log, order=arima_order, trend="n",
                    enforce_stationarity=False, enforce_invertibility=False).fit()

# Residuals for LSTM window
arima_pred_log = np.asarray(
    arima_model.predict(start=train_log.index[0], end=train_log.index[-1], typ="levels")
)
residuals = train_log.values - arima_pred_log
res_scaled = scaler.transform(residuals.reshape(-1, 1)).flatten()

# Take last seq_len residuals
window = res_scaled[-seq_len:].copy().reshape(1, seq_len, 1)

# Load LSTM and predict next residual (scaled)
lstm = load_model(LSTM_PATH, compile=False)
pred_scaled = lstm.predict(window, verbose=0)[0, 0]
pred_residual = scaler.inverse_transform([[pred_scaled]])[0, 0]

# ARIMA next-step forecast
arima_next_log = arima_model.get_forecast(steps=1).predicted_mean.values[0]

# Combine and back-transform
forecast_log = arima_next_log + pred_residual
anchor = np.exp(train_log.iloc[-1])
forecast_price = float(np.exp(forecast_log))

# Optional: anchor to last known price to avoid drift on first step
forecast_price = forecast_price * (anchor / forecast_price)

print(f"Next-day price forecast: {forecast_price:.2f}")
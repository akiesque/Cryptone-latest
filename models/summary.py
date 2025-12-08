
"""
Evaluate the saved hybrid model artifacts (ARIMA order + LSTM .keras + scaler meta)
on a holdout slice of the price series, reporting RMSE, MAE, MAPE, R^2,
and a simple "pseudo accuracy" = 1 - MAPE.

Adjust the paths and HOLDOUT_DAYS as needed, then run:
    python models/summary.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model

# ---- Paths (update if you trained with a different asset) ----
DATA_PATH = Path("dataset/price/btc_prices_2020_present.csv")
LSTM_PATH = Path("models/btc_hybrid_lstm.keras")
META_PATH = Path("models/btc_hybrid_meta.pkl")  # rename to eth_hybrid_meta.pkl if you change it

# How many most-recent days to hold out for evaluation
HOLDOUT_DAYS = 30


def load_meta(meta_path: Path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return (
        tuple(meta["arima_order"]),
        meta["residual_scaler"],
        int(meta["seq_len"]),
    )


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["price"])
    df = df.set_index("timestamp")
    daily = df["price"].resample("D").last().dropna()
    return daily


def forecast_with_artifacts(train_log: pd.Series, forecast_steps: int, arima_order, scaler, seq_len, lstm_path: Path):
    arima_model = ARIMA(train_log, order=arima_order, trend="n", enforce_stationarity=False, enforce_invertibility=False).fit()
    arima_pred_log = np.asarray(arima_model.predict(start=train_log.index[0], end=train_log.index[-1], typ="levels"))
    residuals = train_log.values - arima_pred_log
    res_scaled = scaler.transform(residuals.reshape(-1, 1)).flatten()

    lstm = load_model(lstm_path, compile=False)

    future_residuals = []
    window = res_scaled[-seq_len:].copy()
    for _ in range(forecast_steps):
        lstm_in = window.reshape(1, seq_len, 1)
        pred_scaled = lstm.predict(lstm_in, verbose=0)[0, 0]
        pred_res = scaler.inverse_transform([[pred_scaled]])[0, 0]
        future_residuals.append(pred_res)
        window = np.append(window[1:], pred_scaled)

    future_residuals = np.array(future_residuals)
    future_arima_log = arima_model.get_forecast(steps=forecast_steps).predicted_mean.values
    forecast_log = future_arima_log + future_residuals
    forecast_price = np.exp(forecast_log)

    # anchor first step to last observed price to avoid scale drift on step 1
    anchor = np.exp(train_log.iloc[-1])
    if forecast_price[0] != 0:
        forecast_price = forecast_price * (anchor / forecast_price[0])

    return forecast_price


def mape(true, pred):
    true, pred = np.array(true), np.array(pred)
    eps = 1e-8
    return np.mean(np.abs((true - pred) / (true + eps)))


def main():
    arima_order, scaler, seq_len = load_meta(META_PATH)
    series = load_series(DATA_PATH)

    if len(series) <= HOLDOUT_DAYS + seq_len:
        raise ValueError("Not enough data for the requested holdout window.")

    train = np.log(series.iloc[:-HOLDOUT_DAYS])
    test = np.log(series.iloc[-HOLDOUT_DAYS:])

    fc_price = forecast_with_artifacts(train, forecast_steps=len(test), arima_order=arima_order, scaler=scaler, seq_len=seq_len, lstm_path=LSTM_PATH)
    pred_log = np.log(fc_price[: len(test)])

    rmse = np.sqrt(mean_squared_error(test.values, pred_log))
    mae = mean_absolute_error(test.values, pred_log)
    mape_val = mape(test.values, pred_log)
    r2 = r2_score(test.values, pred_log)
    pseudo_acc = 1 - mape_val

    print(f"Holdout days: {HOLDOUT_DAYS}")
    print(f"RMSE (log): {rmse:.6f}")
    print(f"MAE  (log): {mae:.6f}")
    print(f"MAPE (log): {mape_val:.6f}")
    print(f"R2   (log): {r2:.6f}")
    print(f"Pseudo accuracy (1 - MAPE): {pseudo_acc:.6f}")


if __name__ == "__main__":
    main()

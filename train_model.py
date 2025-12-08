from __future__ import annotations

import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential, load_model

DATA_PATH = Path("dataset/price/xrp_prices_2024_present_1h.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Reduce TF info/warning noise on CPU runs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Silence noisy ARIMA convergence warnings during grid search/forecast fits
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_series(path: Path = DATA_PATH, start: str = "2025-06-01", freq: str = "H") -> pd.Series:
    """Load price CSV, normalize to specified frequency (last value per bin), return series indexed by timestamp."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").dropna(subset=["price"])
    df = df.set_index("timestamp")
    df = df.loc[df.index >= pd.Timestamp(start)]

    # Resample to the target frequency using the last observed price in each bin
    series = df["price"].resample(freq).last()

    # Ensure a regular index with explicit freq to avoid statsmodels warnings
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_index).ffill()
    series.index.freq = to_offset(freq)
    return series


def chronological_split(series: pd.Series, train_ratio=0.7, val_ratio=0.15) -> Tuple[pd.Series, pd.Series, pd.Series]:
    n = len(series)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = series.iloc[:n_train]
    val = series.iloc[n_train : n_train + n_val]
    test = series.iloc[n_train + n_val :]
    return train, val, test


def grid_arima(train_log: pd.Series, p_range=range(0, 4), q_range=range(0, 4)) -> Tuple[Tuple[int, int, int], ARIMA]:
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in p_range:
        for q in q_range:
            try:
                m = ARIMA(
                    train_log,
                    order=(p, 1, q),
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()
                if m.aic < best_aic:
                    best_aic = m.aic
                    best_order = (p, 1, q)
                    best_model = m
            except Exception:
                continue
    if best_model is None:
        raise RuntimeError("ARIMA grid search failed")
    return best_order, best_model


def build_lstm(seq_len: int) -> Sequential:
    model = Sequential(
        [
            Input(shape=(seq_len, 1)),
            LSTM(64, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


@dataclass
class HybridArtifacts:
    arima_order: Tuple[int, int, int]
    residual_scaler: MinMaxScaler
    seq_len: int
    lstm_path: Path
    last_log_price: float
    last_date: pd.Timestamp
    freq: str = "H"


def fit_hybrid(train_log: pd.Series, seq_len: int = 16, lstm_epochs: int = 80, lstm_batch: int = 32, freq: str = "H"):
    # ARIMA
    arima_order, arima_model = grid_arima(train_log)
    arima_pred_log = np.asarray(arima_model.predict(start=train_log.index[0], end=train_log.index[-1], typ="levels"))
    residuals = train_log.values - arima_pred_log

    # LSTM on residuals
    scaler = MinMaxScaler()
    res_scaled = scaler.fit_transform(residuals.reshape(-1, 1)).flatten()
    X, y = [], []
    for i in range(seq_len, len(res_scaled)):
        X.append(res_scaled[i - seq_len : i])
        y.append(res_scaled[i])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)

    val_size = max(1, int(0.2 * len(X)))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    lstm = build_lstm(seq_len)
    callbacks = [EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
    lstm.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=lstm_epochs,
        batch_size=lstm_batch,
        verbose=1,  
        callbacks=callbacks,
    )

    lstm_path = MODELS_DIR / "xrp_hybrid_lstm.keras"
    lstm.save(lstm_path)

    artifacts = HybridArtifacts(
        arima_order=arima_order,
        residual_scaler=scaler,
        seq_len=seq_len,
        lstm_path=lstm_path,
        last_log_price=train_log.iloc[-1],
        last_date=train_log.index[-1],
        freq=freq,
    )
    return artifacts, arima_pred_log, residuals


def forecast_hybrid(artifacts: HybridArtifacts, train_log: pd.Series, forecast_steps: int = 120, noise_scale: float = 0.05):
    arima_model = ARIMA(train_log, order=artifacts.arima_order, trend="n", enforce_stationarity=False, enforce_invertibility=False).fit()
    future_arima_log = arima_model.get_forecast(steps=forecast_steps).predicted_mean.values

    # Load without compiling to avoid deserializing legacy metrics/optimizer from H5
    lstm = load_model(artifacts.lstm_path, compile=False)
    arima_pred_log = np.asarray(arima_model.predict(start=train_log.index[0], end=train_log.index[-1], typ="levels"))
    residuals = train_log.values - arima_pred_log
    res_scaled = artifacts.residual_scaler.transform(residuals.reshape(-1, 1)).flatten()

    seq_len = artifacts.seq_len
    future_residuals = []
    window = res_scaled[-seq_len:].copy()
    for _ in range(forecast_steps):
        lstm_in = window.reshape(1, seq_len, 1)
        pred_scaled = lstm.predict(lstm_in, verbose=0)[0, 0]
        pred_res = artifacts.residual_scaler.inverse_transform([[pred_scaled]])[0, 0]
        future_residuals.append(pred_res)
        window = np.append(window[1:], pred_scaled)

    future_residuals = np.array(future_residuals)
    noise = np.random.normal(0, residuals.std() * noise_scale, size=future_residuals.shape)
    forecast_log = future_arima_log + future_residuals + noise
    forecast_price = np.exp(forecast_log)
    freq_offset = to_offset(getattr(artifacts, "freq", "H"))
    forecast_index = pd.date_range(start=train_log.index[-1] + freq_offset, periods=forecast_steps, freq=freq_offset)

    if forecast_price[0] > 0:
        anchor = np.exp(train_log.iloc[-1])
        forecast_price = forecast_price * (anchor / forecast_price[0])

    return forecast_index, forecast_price


def evaluate_series(true: np.ndarray, pred: np.ndarray):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    return rmse, mae


def mape(true: np.ndarray, pred: np.ndarray) -> float:
    true_arr, pred_arr = np.array(true), np.array(pred)
    eps = 1e-8
    return np.mean(np.abs((true_arr - pred_arr) / (true_arr + eps)))


def compute_metrics(true: np.ndarray, pred: np.ndarray):
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape_val = mape(true, pred)
    pseudo_acc = 1 - mape_val
    return rmse, mae, r2, mape_val, pseudo_acc


def walk_forward_eval(series: pd.Series, forecast_horizon: int = 30, train_min: int = 200):
    """Simple walk-forward: expand train window, forecast on next segment."""
    results = []
    log_series = np.log(series)
    dates = log_series.index
    for split_idx in range(train_min, len(log_series) - forecast_horizon, forecast_horizon):
        train_log = log_series.iloc[:split_idx]
        true_segment = log_series.iloc[split_idx : split_idx + forecast_horizon]
        artifacts, _, _ = fit_hybrid(train_log, seq_len=16, lstm_epochs=5, lstm_batch=16, freq=series.index.freqstr or "H")
        _, fc_price = forecast_hybrid(artifacts, train_log, forecast_steps=forecast_horizon, noise_scale=0.02)
        true_segment_price = np.exp(true_segment.values)
        pred_segment_price = fc_price[: len(true_segment)]
        rmse, mae, r2, mape_val, pacc = compute_metrics(true_segment_price, pred_segment_price)
        results.append((dates[split_idx], rmse, mae, r2, mape_val, pacc))
    return results


def run_pipeline():
    series = load_series()
    log_series = np.log(series)
    train, val, test = chronological_split(log_series)

    freq = series.index.freqstr or "H"

    artifacts, _, _ = fit_hybrid(train, seq_len=16, lstm_epochs=60, lstm_batch=32, freq=freq)
    val_idx, val_fc = forecast_hybrid(artifacts, train, forecast_steps=len(val), noise_scale=0.02)
    val_true = np.exp(val.values)
    val_pred = val_fc[: len(val)]
    val_rmse, val_mae, val_r2, val_mape, val_pacc = compute_metrics(val_true, val_pred)

    artifacts_full, _, _ = fit_hybrid(pd.concat([train, val]), seq_len=16, lstm_epochs=60, lstm_batch=32, freq=freq)
    test_idx, test_fc = forecast_hybrid(artifacts_full, pd.concat([train, val]), forecast_steps=len(test), noise_scale=0.02)
    test_true = np.exp(test.values)
    test_pred = test_fc[: len(test)]
    test_rmse, test_mae, test_r2, test_mape, test_pacc = compute_metrics(test_true, test_pred)

    print("Validation metrics:")
    print(f"  RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f} | R2: {val_r2:.6f} | MAPE: {val_mape:.6f} | Pseudo Acc: {val_pacc:.6f}")
    print("Test metrics:")
    print(f"  RMSE: {test_rmse:.6f} | MAE: {test_mae:.6f} | R2: {test_r2:.6f} | MAPE: {test_mape:.6f} | Pseudo Acc: {test_pacc:.6f}")

    meta = {
        "arima_order": artifacts_full.arima_order,
        "seq_len": artifacts_full.seq_len,
        "residual_scaler": artifacts_full.residual_scaler,
        "last_log_price": artifacts_full.last_log_price,
        "last_date": str(artifacts_full.last_date),
        "freq": artifacts_full.freq,
    }
    with open(MODELS_DIR / "xrp_hybrid_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    wf_results = walk_forward_eval(series, forecast_horizon=30, train_min=200)
    if wf_results:
        avg_rmse = np.mean([r[1] for r in wf_results])
        avg_mae = np.mean([r[2] for r in wf_results])
        avg_r2 = np.mean([r[3] for r in wf_results])
        avg_mape = np.mean([r[4] for r in wf_results])
        avg_pacc = np.mean([r[5] for r in wf_results])
        print("Walk-forward averages:")
        print(f"  RMSE: {avg_rmse:.6f} | MAE: {avg_mae:.6f} | R2: {avg_r2:.6f} | MAPE: {avg_mape:.6f} | Pseudo Acc: {avg_pacc:.6f}")


if __name__ == "__main__":
    run_pipeline()


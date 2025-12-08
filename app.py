import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import base64
import time
import pickle
from pathlib import Path

from sample_sentiment import get_mock_sentiment

from call_model import load_predmodel, call_data, predict_next_price

from binance.client import Client
from requests.exceptions import ConnectionError
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from pandas.tseries.frequencies import to_offset

show = True

## Binance API 
API_KEY = st.secrets["BINANCE_API_KEY"]
API_SECRET = st.secrets["BINANCE_API_SECRET"]

try:
    client = Client(API_KEY, API_SECRET)
except ConnectionError:
    pass

## functions


def get_base64_img(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# st.button("Sample")

# Streamlit app title
st.title("Cryptone")

st.markdown("""
<style>
.button {
    display: inline-block;
    padding: 0.6em 1.2em;
    margin: 0.5em;
    font-size: 1.1em;
    color: white;
    background-color: #444;
    border-radius: 10px;
    text-decoration: none;
    transition: all 0.2s ease-in-out;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}
.button:hover {
    background-color: #666;
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

## switch to 'BTCPHP'

# List of coins
coins = [
    {"name": "Bitcoin", "symbol": "BTC", "img": './static/bitcoin-btc-logo.png', "id": "BTCUSDT", "label": "BTC/USDT", "model": "btc"},
    {"name": "Ethereum", "symbol": "ETH", "img": './static/ethereum-eth-logo.png', "id": "ETHUSDT", "label": "ETH/USDT",  "model": "eth"},
    {"name": "XRP", "symbol": "XRP", "img": './static/xrp-xrp-logo.png', "id": "XRPUSDT", "label": "XRP/USDT",  "model": "xrp"},
    # {"name": "Tether", "symbol": "USDT", "img": './static/tether-usdt-logo.png', "id": "DOGEUSDT", "label": "USDT Conversion Rate"}
]

# Sidebar title
with st.sidebar:
    st.markdown("### Select Cryptocurrency")
    cols = st.columns(2)
    for i, coin in enumerate(coins):
        with cols[i % 2]:
            if st.button(coin["name"], key=coin["symbol"]):
                st.session_state.selected_coin = coin["symbol"]
            # if st.button:
            #     msg = st.empty()
            #     st.success(f"You selected: {coin['name']}")
            #     msg.empty()
            #     time.sleep(3)

## sets BTC as default.
selected_coin = st.session_state.get("selected_coin", coins[0].get('symbol'))

coin = next((c for c in coins if c["symbol"] == selected_coin), coins[0])

# st.header(coin.get("name"))

st.markdown("""
    <style>
    .coin-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding-top: 40px;
    }
    .coin-image {
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .coin-image:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

selected_coin_data = next((coin for coin in coins if coin["symbol"] == selected_coin), coins[0])

# ## variables
# scaled_features, scaler, df = call_data(f"{coin.get("model")}")
# model = load_predmodel(f"{coin.get("model")}")

# predicted = predict_next_price(model, scaled_features)

# Display current price
# current_price = df[f'price_{coin.get("model")}'].values[-1]

col1, col2 = st.columns([1,3])

with col1:
    if coin:
        img_base64 = get_base64_img(coin["img"])
        st.markdown(
            f"""
            <div class='coin-container'>
                <img src="data:image/png;base64,{img_base64}" width="150" class="coin-image"/>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
     # Auto-refresh every 10 seconds
    st_autorefresh(interval=5000, key="refresh")

    if show:

        try:
            # # Auto-refresh every 10 seconds
            # st_autorefresh(interval=10000, key="refresh")

            # Get price
            price_data = client.get_symbol_ticker(symbol=coin.get("id"))
            current_price = float(price_data["price"])

            # Delta logic with session state
            if "last_price" not in st.session_state:
                st.session_state.last_price = {}

            last_price = st.session_state.last_price.get(selected_coin)
            price_change = current_price - last_price if last_price is not None else 0.0
            delta = f"{price_change:+.6f}" if last_price is not None else "↺"

            st.session_state.last_price[selected_coin] = current_price

            # Display stylized markdown for price + delta
            st.markdown(
                f"""
                <div style="
                    padding-bottom: 10px;">
                <div style="
                    background-color: #1e1e1e;
                    padding: 1rem;
                    border-radius: 1rem;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
                    color: white;
                ">
                    <h3 style="margin: 0;">{coin.get("name")}</h3>
                    <p style="margin: 0; font-size: 0.9rem; color: #888;">{coin.get("label")}</p>
                    <h1 style="margin: 0; color: #00FFAA; font-size: 2rem;">
                        {current_price:.6f}
                    </h1>
                    <p style="margin: 0; font-size: 1rem; color: {'#00ff00' if price_change >= 0 else '#ff5555'};">
                        {delta}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except ConnectionError:
            pass

        except Exception as e:
            st.error(f'''If you're seeing this message, that just means you're coding at school u sick deranged woman 
                    aren't u aware binance don't work outside cuz starlink diff? Well now you know.''')

    
# sentiment = get_mock_sentiment() ## bullish, high, low... etc
# predicted_price = predict_next_price()

# df = pd.read_csv("C:/Users/steph/Desktop/MIMI/Jupynotebooks/ThesisTrial/empathic - binance ver/dataset/xrp_sentiment_scored.csv")

# # Load LSTM Model
# model = tf.keras.models.load_model("C:/Users/steph/Desktop/MIMI/Jupynotebooks/ThesisTrial/empathic - binance ver/model/xrp_lstm_model.h5")

# # Predict Next Price (Basic Example)
# def predict_next_price():
#     last_prices = df['price_xrp'].values[-5:]
#     last_prices_scaled = np.array(last_prices).reshape(1, -1, 1)
#     prediction = model.predict(last_prices_scaled)
#     return round(float(prediction[0][0]), 2)

# sentiment = df["compound"].values[-1] 


# style for the card

st.markdown("""
<style>
.card {
    padding: 1px;
    margin-bottom: 20px;
    background-color: #0077AB;
    border-radius: 12px;
    text-align: center;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.05);
}
.metric-label {
    font-size: 1.2rem;
    color: #0001E;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Display metrics in columns
col1, col2, col3 = st.columns(3)

## stuffs
# df = pd.read_csv(f"./dataset/{coin.get("model")}_sentiment_scored.csv")
# sen_score = df['compound'].iloc[5] if not df.empty and 'compound' in df.columns else 0 
# sentiment = df['sentiment'].iloc[-1] if not df.empty and 'sentiment' in df.columns else "Neutral" 

# # Load predicted prices
# pred_df = pd.read_csv(f"./dataset/{coin.get("model")}_predicted_prices.csv")
# pred_df["date"] = pd.to_datetime(pred_df["date"])

# with col1:
#     st.markdown(f"""
#     <div class="card">
#         <div class="metric-label">📉 Sentiment Score</div>
#         <div class="metric-value">Placeholder</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown(f"""
#     <div class="card">
#         <div class="metric-label">🙎 Emotion</div>
#         <div class="metric-value">Placeholder</div>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown(f"""
#     <div class="card">
#         <div class="metric-label">🔮 Predicted Price</div>
#         <div class="metric-value">Placeholder</div>
#     </div>
#     """, unsafe_allow_html=True)

# Load and prepare historical data
df = pd.read_csv(f"./dataset/price/{coin.get("model")}_prices_2020_present.csv")  # Update path
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df.set_index("timestamp", inplace=True)

# Load and prepare predicted data
pred_df = pd.read_csv(f"./outputs/{coin.get("model")}_hybrid_forecast.csv")  # Should contain columns: date, predicted_price
pred_df["date"] = pd.to_datetime(pred_df["date"])
pred_df = pred_df.sort_values("date")
pred_df.set_index("date", inplace=True)

# Load and prepare historically predicted data
hist_pred = pd.read_csv(f"./outputs/{coin.get("model")}_hybrid_in_sample.csv") # Should contain columns: date, predicted_price
hist_pred['date'] = pd.to_datetime(hist_pred['date'])
hist_pred = hist_pred.sort_values('date')
hist_pred.set_index("date", inplace=True)

@st.cache_resource
def load_hybrid_artifacts(model_key: str):
    meta_path = Path(f"models/{model_key}_hybrid_meta.pkl")
    lstm_path = Path(f"models/{model_key}_hybrid_lstm.keras")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    lstm = load_model(lstm_path, compile=False)
    return meta, lstm


def load_series_for_coin(model_key: str, start: str | None, freq: str):
    path = Path(f"./dataset/price/{model_key}_prices_2020_present.csv")
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").dropna()
    df = df.set_index("timestamp")
    price_col = "price"
    if f"price_{model_key}" in df.columns:
        price_col = f"price_{model_key}"
    series = df[price_col]
    if start:
        series = series.loc[series.index >= pd.Timestamp(start)]
    series = series.resample(freq).last().ffill()
    series.index.freq = to_offset(freq)
    return series


def predict_next_with_hybrid(model_key: str, start: str | None = None):
    meta, lstm = load_hybrid_artifacts(model_key)
    arima_order = tuple(meta["arima_order"])
    scaler = meta["residual_scaler"]
    seq_len = int(meta["seq_len"])
    freq = meta.get("freq", "H")

    series = load_series_for_coin(model_key, start=start, freq=freq)
    log_series = np.log(series)

    arima_model = ARIMA(log_series, order=arima_order, trend="n", enforce_stationarity=False, enforce_invertibility=False).fit()
    arima_pred_log = np.asarray(arima_model.predict(start=log_series.index[0], end=log_series.index[-1], typ="levels"))
    residuals = log_series.values - arima_pred_log
    res_scaled = scaler.transform(residuals.reshape(-1, 1)).flatten()

    window = res_scaled[-seq_len:].copy().reshape(1, seq_len, 1)
    pred_scaled = lstm.predict(window, verbose=0)[0, 0]
    pred_res = scaler.inverse_transform([[pred_scaled]])[0, 0]

    arima_next_log = arima_model.get_forecast(steps=1).predicted_mean.values[0]
    forecast_log = arima_next_log + pred_res
    forecast_price = float(np.exp(forecast_log))
    anchor = float(series.iloc[-1])
    if forecast_price > 0:
        forecast_price = forecast_price * (anchor / forecast_price)

    next_time = series.index[-1] + to_offset(freq)
    return forecast_price, next_time, series.tail(100)


tab1, tab2= st.tabs(["📈Historical Price Forecast", f"💬 LIVE | {coin.get("name")} Predicted Price"])

with tab1:
    # Create the plot
    fig = go.Figure()

    # Add historical prices
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[f"price_{coin.get("model")}"],
        mode="lines",
        name="Historical Price",
        line=dict(color="royalblue")
    ))

    # Add predicted prices (make sure dates are after the historical range)
    fig.add_trace(go.Scatter(
        x=pred_df.index,
        y=pred_df[f"{coin.get("model")}_forecast_price"],
        mode="lines",
        name="Predicted Price",
        line=dict(color="orange", dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=hist_pred.index,
        y=hist_pred[f"{coin.get("model")}_hybrid_pred"],
        mode="lines",
        name="Model Prediction (Training)",
        line=dict(color="orange", dash="dot"),
        opacity= 0.5
    ))

    # Layout adjustments
    fig.update_layout(
        title=f"📈 {coin.get("name")} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (in USD)",
        hovermode="x unified",
        hoverlabel=dict(
            font_size=16,
            font_family="Arial"
        ),
        showlegend=True
    )

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.badge("Live short-horizon forecast is not exactly accurate. Use as reference only.", icon=":material/question_mark:", color="orange")

    current_display = f"{current_price:.6f}" if "current_price" in locals() else "N/A"
    st.metric("Current price", current_display)

    with st.spinner("Loading model and generating forecast..."):
        try:
            next_pred_price, next_pred_time, recent_series = predict_next_with_hybrid(coin.get("model"))
        except Exception as e:
            next_pred_price, next_pred_time, recent_series = None, None, None
            st.error(f"Live forecast unavailable: {e}")

    if next_pred_price is not None:
        horizon_delta = (next_pred_time - recent_series.index.max())
        horizon_minutes = int(horizon_delta.total_seconds() // 60)
        horizon_label = f"~{horizon_minutes} minutes" if horizon_minutes < 180 else f"~{horizon_delta}"
        delta_val = next_pred_price - current_price if "current_price" in locals() else None
        st.metric(f"Forecast ({horizon_label})", f"{next_pred_price:.6f}", delta=f"{delta_val:+.6f}" if delta_val is not None else None)

        recent_df = recent_series.reset_index()
        next_point = pd.DataFrame({"timestamp": [next_pred_time], "price": [next_pred_price]})
        chart_df = pd.concat(
            [
                recent_df.rename(
                    columns={recent_df.columns[0]: "timestamp", recent_df.columns[1]: "price"}
                ),
                next_point,
            ],
            ignore_index=True,
        )
        fig_live = px.line(chart_df, x="timestamp", y="price", title="Recent price | next forecast", markers=True)
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.info("Forecast data not available for this coin.")
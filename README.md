
# Cryptone

Welcome to Cryptone: a Web Application Prototype UI that predicted historical prices of three coins: Bitcoin (BTC), Ethereum (ETH) and Ripple (XRP).

---
### Features:
- Uses ARIMA-LSTM models for price volatility, and machine learning integration
- Interactive Streamlit UI
- Supports Binance integration (API keys are securely handled through a .toml)

---
### How to use:
- Clone this repo:
    ```bash
   git clone https://github.com/akiesque/Cryptone.git
   cd Cryptone
   
- Install dependencies (Python 3.10+, etc,..)
- Create your own `.streamlit/secrets.toml` file using your own Binance API key. For CryptoPanic, simply replace the `your_auth_token` in the `emotion_model.py` file.
- Once it's done, you can run the app through `streamlit run app.py`
---
### A few notes...
- Binance's website cannot be accessed in web browsers when under Philippine Internet Service Providers or telco.
    - We advise you to look into local alternatives *(CoinGecko, coins.ph)* but make sure to read their documentation. Code may change accordingly.
- To create your Binance API key: go to your account > profile > API Management (this can only be accessed through browser)
---
this project was made by two developers who are inspired by machine learning and crypto trading. happy trading to everyone 💕
### Thank you 💖 



## Authors

- [@Brrrt2](https://github.com/Brrrt2) -- the machine learning person
- [@akiesque](https://github.com/akiesque) -- UI/UX

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# -----------------------------
# 1. Ambil data dari yfinance
# -----------------------------
def ambil_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[['Close']].asfreq('B').fillna(method='ffill')
    data.dropna(inplace=True)
    return data

# -----------------------------
# 2. Bangun model ARIMA
# -----------------------------
def bangun_model_arima(data, order=(3,1,2), hari_prediksi=10):
    if data.empty or 'Close' not in data:
        return None

    try:
        model = ARIMA(data['Close'], order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=hari_prediksi)
        return forecast
    except Exception as e:
        st.error(f"❌ Gagal membangun model ARIMA: {e}")
        return None

# -----------------------------
# 3. Gabungkan data aktual + prediksi
# -----------------------------
def gabungkan_data(data, forecast):
    if forecast is None:
        return data, pd.DataFrame()

    last_date = data.index[-1]
    pred_dates = pd.bdate_range(last_date + timedelta(1), periods=len(forecast))
    df_forecast = pd.DataFrame(forecast.values, index=pred_dates, columns=["Close"])
    df_forecast.index.name = 'Date'
    data_gabungan = pd.concat([data[['Close']], df_forecast], axis=0)
    return data_gabungan, df_forecast

# -----------------------------
# 4. Visualisasi Harga + Prediksi
# -----------------------------
def plot_prediksi(data, forecast, symbol):
    if forecast is None or data.empty:
        st.warning("Tidak ada data untuk ditampilkan.")
        return

    last_date = data.index[-1]
    pred_dates = pd.bdate_range(last_date + timedelta(1), periods=len(forecast))

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(data.index, data['Close'], label='Harga Aktual', marker='o', color='blue')
    ax.plot(pred_dates, forecast, label='Prediksi ARIMA (10 Hari)', marker='x', linestyle='--', color='orange')

    ax.set_title(f"📈 Harga Aktual & Prediksi ARIMA — {symbol}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga Penutupan")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# 5. Analisis Fuzzy
# -----------------------------
def analisis_fuzzy(data_gabungan, symbol):
    if data_gabungan.empty:
        st.warning("Data gabungan kosong, tidak dapat melakukan analisis fuzzy.")
        return

    data = data_gabungan.copy()
    data['Return'] = data['Close'].pct_change() * 100
    data.dropna(inplace=True)

    harga = ctrl.Antecedent(np.linspace(data['Close'].min(), data['Close'].max(), 100), 'harga')
    perubahan = ctrl.Antecedent(np.linspace(data['Return'].min(), data['Return'].max(), 100), 'perubahan')
    sinyal = ctrl.Consequent(np.arange(0, 101, 1), 'sinyal')

    harga['rendah'] = fuzz.trimf(harga.universe, [data['Close'].min(), data['Close'].min(), data['Close'].mean()])
    harga['normal'] = fuzz.trimf(harga.universe, [data['Close'].min(), data['Close'].mean(), data['Close'].max()])
    harga['tinggi'] = fuzz.trimf(harga.universe, [data['Close'].mean(), data['Close'].max(), data['Close'].max()])

    perubahan['turun'] = fuzz.trimf(perubahan.universe, [data['Return'].min(), data['Return'].min(), 0])
    perubahan['stabil'] = fuzz.trimf(perubahan.universe, [data['Return'].min(), 0, data['Return'].max()])
    perubahan['naik'] = fuzz.trimf(perubahan.universe, [0, data['Return'].max(), data['Return'].max()])

    sinyal['jual'] = fuzz.trimf(sinyal.universe, [0, 0, 50])
    sinyal['tahan'] = fuzz.trimf(sinyal.universe, [25, 50, 75])
    sinyal['beli'] = fuzz.trimf(sinyal.universe, [50, 100, 100])

    rules = [
        ctrl.Rule(harga['rendah'] & perubahan['naik'], sinyal['beli']),
        ctrl.Rule(harga['normal'] & perubahan['stabil'], sinyal['tahan']),
        ctrl.Rule(harga['tinggi'] & perubahan['turun'], sinyal['jual']),
        ctrl.Rule(perubahan['naik'] & harga['tinggi'], sinyal['tahan']),
        ctrl.Rule(perubahan['turun'] & harga['rendah'], sinyal['tahan'])
    ]

    sinyal_ctrl = ctrl.ControlSystem(rules)
    simulator = ctrl.ControlSystemSimulation(sinyal_ctrl)

    fuzzy_values, decisions = [], []
    for i in range(len(data)):
        simulator.input['harga'] = data['Close'].iloc[i]
        simulator.input['perubahan'] = data['Return'].iloc[i]
        simulator.compute()
        nilai = simulator.output['sinyal']
        fuzzy_values.append(nilai)
        if nilai < 33:
            decisions.append("JUAL")
        elif nilai < 66:
            decisions.append("TAHAN")
        else:
            decisions.append("BELI")

    data["FuzzySignal"] = fuzzy_values
    data["Decision"] = decisions

    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(data.index, data['Close'], label='Harga', color='blue')
    ax.plot(data.index, data['FuzzySignal'], label='Nilai Fuzzy (0–100)', color='cyan', linewidth=2)
    ax.axhspan(0, 33, color='red', alpha=0.1)
    ax.axhspan(33, 66, color='yellow', alpha=0.1)
    ax.axhspan(66, 100, color='green', alpha=0.1)
    ax.set_title(f"📊 Sinyal Fuzzy Logic: {symbol}")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga & Nilai Fuzzy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

    latest = data.iloc[-1]
    st.markdown(f"""
    ### 📌 Analisis Terakhir ({symbol})
    - 💰 Harga: **{latest['Close']:.2f}**
    - 📉 Perubahan: **{latest['Return']:.2f}%**
    - 🎚️ Nilai Fuzzy: **{latest['FuzzySignal']:.2f}**
    - 🟢 Rekomendasi: **{latest['Decision']}**
    """)

    st.dataframe(data.tail(10))

# -----------------------------
# 6. UI Streamlit
# -----------------------------
st.set_page_config(page_title="ARIMA + Fuzzy Stock Forecast", layout="wide")

st.title("📈 Prediksi Harga & Analisis Fuzzy")
symbol = st.selectbox("Pilih Aset:", {
    "Gold Futures": "GC=F",
    "Bitcoin (USD)": "BTC-USD",
    "Crude Oil WTI": "CL=F",
    "S&P 500": "^GSPC",
    "Ethereum (ETH)": "ETH-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Pepe Coin (PEPE)": "PEPE-USD",
    "Ripple (XRP)": "XRP-USD",
    "Solana (SOL)": "SOL-USD"
})

periode = st.selectbox("Pilih Periode:", ["1mo","3mo","6mo","1y","2y","5y"])
interval = st.selectbox("Pilih Interval:", ["1d", "1wk"])

if st.button("🔮 Jalankan Analisis"):
    data = ambil_data(symbol, periode, interval)
    if data.empty:
        st.error("❌ Tidak ada data untuk simbol dan periode ini.")
    else:
        forecast = bangun_model_arima(data)
        if forecast is not None:
            data_gabungan, df_prediksi = gabungkan_data(data, forecast)
            plot_prediksi(data, forecast, symbol)
            analisis_fuzzy(data_gabungan, symbol)

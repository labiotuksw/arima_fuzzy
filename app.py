import yfinance as yf
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import streamlit as st

# -----------------------------
# Konfigurasi Halaman
# -----------------------------
st.set_page_config(page_title="ARIMA + Fuzzy Logic Forecast", layout="wide")

st.title("📊 ARIMA + Fuzzy Logic Stock Forecasting")
st.markdown("""
Aplikasi ini memprediksi harga **saham/crypto** menggunakan model **ARIMA** dan 
memberikan rekomendasi **Beli / Tahan / Jual** berdasarkan **Logika Fuzzy**.
""")

# -----------------------------
# 1. Ambil data dari yfinance
# -----------------------------
def ambil_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Close']].asfreq('B').fillna(method='ffill')
    return data

# -----------------------------
# 2. Bangun model ARIMA
# -----------------------------
def bangun_model_arima(data, order=(3,1,2), hari_prediksi=10):
    model = ARIMA(data['Close'], order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=hari_prediksi)
    return forecast

# -----------------------------
# 3. Gabungkan data aktual + prediksi
# -----------------------------
def gabungkan_data(data, forecast):
    last_date = data.index[-1]
    pred_dates = pd.bdate_range(last_date + timedelta(1), periods=len(forecast))
    df_forecast = pd.DataFrame(forecast.values, index=pred_dates, columns=["Close"])
    df_forecast.index.name = 'Date'
    data_gabungan = pd.concat([data[['Close']], df_forecast], axis=0)
    return data_gabungan, df_forecast

# -----------------------------
# 4. Analisis Fuzzy
# -----------------------------
def analisis_fuzzy(data_gabungan):
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
        elif nilai < 40:
            decisions.append("TAHAN")
        else:
            decisions.append("BELI")

    data["FuzzySignal"] = fuzzy_values
    data["Decision"] = decisions
    return data

# -----------------------------
# 5. Streamlit UI
# -----------------------------
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("📈 Pilih Aset", {
        "GC=F": "Gold Futures",
        "BTC-USD": "Bitcoin (USD)",
        "CL=F": "Crude Oil WTI",
        "^GSPC": "S&P 500",
        "ETH-USD": "Ethereum (ETH)",
        "DOGE-USD": "Dogecoin (DOGE)",
        "PEPE-USD": "Pepe Coin (PEPE)",
        "XRP-USD": "Ripple (XRP)",
        "SOL-USD": "Solana (SOL)",
        "BBCA.JK": "Bank Central Asia",
        "BBRI.JK": "Bank Rakyat Indonesia",
        "BMRI.JK": "Bank Mandiri",
        "TLKM.JK": "Telkom Indonesia",
        "UNVR.JK": "Unilever Indonesia",
        "ASII.JK": "Astra International",
        "ICBP.JK": "Indofood CBP",
        "INDF.JK": "Indofood Sukses Makmur",
        "KLBF.JK": "Kalbe Farma",
        "ANTM.JK": "Aneka Tambang",
        "INCO.JK": "Vale Indonesia",
        "ADRO.JK": "Adaro Energy",
        "PTBA.JK": "Bukit Asam",
        "CPIN.JK": "Charoen Pokphand",
        "ITMG.JK": "Indo Tambangraya Megah",
        "ARTO.JK": "Bank Jago",
        "BFIN.JK": "BFI Finance",
        "GOTO.JK": "GoTo Gojek Tokopedia",
        "MEGA.JK": "Bank Mega"
    })
with col2:
    periode = st.selectbox("🕒 Periode Data", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=4)
with col3:
    interval = st.selectbox("⏳ Interval", ["1d", "1wk"], index=0)

# -----------------------------
# 6. Jalankan Analisis
# -----------------------------
if st.button("🚀 Jalankan Analisis"):
    with st.spinner("Mengambil data dan menjalankan model..."):
        data = ambil_data(symbol, periode, interval)
        forecast = bangun_model_arima(data)
        data_gabungan, df_prediksi = gabungkan_data(data, forecast)
        df_fuzzy = analisis_fuzzy(data_gabungan)

        # -----------------------------
        # Tampilkan Grafik
        # -----------------------------
        st.subheader("📈 Grafik Harga Aktual & Prediksi ARIMA")
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(data.index, data['Close'], label='Harga Aktual', color='blue')
        ax.plot(df_prediksi.index, df_prediksi['Close'], label='Prediksi ARIMA', linestyle='--', color='orange')
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)

        # -----------------------------
        # Tabel hasil fuzzy
        # -----------------------------
        st.subheader("📋 Tabel Hasil Analisis Fuzzy Logic")
        st.dataframe(df_fuzzy.tail(10))  # tampilkan 10 terakhir

        # -----------------------------
        # Rekomendasi terakhir
        # -----------------------------
        latest = df_fuzzy.iloc[-1]
        st.markdown("---")
        st.markdown(f"""
        ## 📌 Rekomendasi Terakhir:
        - 💰 **Harga:** {latest['Close']:.2f}
        - 📉 **Perubahan:** {latest['Return']:.2f}%
        - 🎚️ **Nilai Fuzzy:** {latest['FuzzySignal']:.2f}
        - 🟢 **Sinyal:** `{latest['Decision']}`
        """)

        st.success("✅ Analisis selesai! Lihat tabel dan rekomendasi di atas.")

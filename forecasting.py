import streamlit as st
import pandas as pd
import matplotlib.subplots as plt
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Mengabaikan peringatan sistem
warnings.filterwarnings("ignore")

st.title("Sistem Demand Forecasting Persediaan")
st.write("PT Samudra Utama Narapati - Oleh: Sri Devi")
st.markdown("---")

# FITUR BARU: CACHE DATA AGAR WEB SANGAT CEPAT (TIDAK LOADING ULANG SAAT GANTI BARANG)
@st.cache_data
def load_and_clean_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    df = df.dropna(subset=['Item Description', 'Invoice Date'])
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    df['Bulan'] = df['Invoice Date'].dt.to_period('M')
    df_bulanan = df.groupby(['Item Description', 'Bulan'])['Quantity'].sum().reset_index()
    df_bulanan['Bulan'] = df_bulanan['Bulan'].dt.to_timestamp()
    
    return df_bulanan

# 1. Fitur Upload File
uploaded_file = st.file_uploader("Unggah file Penjualan (Excel/CSV)", type=["xls", "xlsx", "csv"])

if uploaded_file is not None:
    with st.spinner('Memproses data... (Hanya butuh waktu di awal)'):
        # Memanggil fungsi cache di atas
        df_bulanan = load_and_clean_data(uploaded_file)

    st.success("Data berhasil dibaca dan siap dianalisis!")

    # 2. Interaksi User: Pilih Barang
    daftar_barang = df_bulanan['Item Description'].unique()
    barang_dipilih = st.selectbox("Pilih Barang untuk Diproyeksikan:", daftar_barang)

    # Filter data sesuai barang yang dipilih secara dinamis
    data_barang = df_bulanan[df_bulanan['Item Description'] == barang_dipilih].sort_values('Bulan')
    
    # Menampilkan grafik historis (Menggunakan f-string agar nama berubah otomatis)
    st.subheader(f"Tren Historis: {barang_dipilih}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data_barang['Bulan'], data_barang['Quantity'], marker='o', linestyle='-', color='blue', label='Data Aktual')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    # 3. Tombol Eksekusi Prediksi ARIMA
    st.markdown("---")
    if st.button("Jalankan Prediksi 12 Bulan Ke Depan"):
        with st.spinner('Algoritma ARIMA sedang menghitung proyeksi...'):
            ts_data = data_barang.set_index('Bulan')['Quantity']
            
            # Memastikan urutan waktu tidak bolong
            ts_data = ts_data.resample('MS').sum().fillna(0)
            
            # Membangun Model ARIMA
            model = ARIMA(ts_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Melakukan peramalan 12 langkah
            forecast = model_fit.forecast(steps=12)
            forecast_index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
            forecast.index = forecast_index

            # Menampilkan Grafik Gabungan
            st.subheader(f"Hasil Peramalan Stok: {barang_dipilih} (Sep 2026 - Ags 2027)")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(ts_data.index, ts_data.values, marker='o', linestyle='-', color='blue', label='Data Aktual')
            ax2.plot(forecast.index, forecast.values, marker='o', linestyle='--', color='red', label='Hasil Prediksi')
            
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            st.pyplot(fig2)

            # Menampilkan tabel hasil
            st.write("**Rincian Angka Proyeksi:**")
            df_forecast = pd.DataFrame({'Bulan': forecast.index.strftime('%B %Y'), 'Prediksi Kebutuhan (Quantity)': forecast.values.round()})
            st.dataframe(df_forecast, use_container_width=True)

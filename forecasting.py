import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Mengabaikan peringatan sistem agar tampilan web tetap bersih
warnings.filterwarnings("ignore")

st.title("Sistem Demand Forecasting Persediaan")
st.write("PT Samudra Utama Narapati - Oleh: Sri Devi")
st.markdown("---")

# 1. Fitur Upload File
uploaded_file = st.file_uploader("Unggah file Excel Penjualan yang sudah dibersihkan", type=["xls", "xlsx", "csv"])

if uploaded_file is not None:
    with st.spinner('Memproses dan merapikan data transaksi...'):
        # Membaca data secara langsung karena file sudah bersih
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Membersihkan baris yang kosong
        df = df.dropna(subset=['Item Description', 'Invoice Date'])
        
        # Menyeragamkan format tanggal dan angka (menggunakan nama kolom baru 'Quantity')
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        
        # Konversi ke agregat bulanan
        df['Bulan'] = df['Invoice Date'].dt.to_period('M')
        df_bulanan = df.groupby(['Item Description', 'Bulan'])['Quantity'].sum().reset_index()
        df_bulanan['Bulan'] = df_bulanan['Bulan'].dt.to_timestamp()

    st.success("Data berhasil dibaca dan diproses!")

    # 3. Interaksi User: Pilih Barang
    daftar_barang = df_bulanan['Item Description'].unique()
    barang_dipilih = st.selectbox("Pilih Barang untuk Diproyeksikan:", daftar_barang)

    # Filter data sesuai barang yang dipilih
    data_barang = df_bulanan[df_bulanan['Item Description'] == barang_dipilih].sort_values('Bulan')
    
    # Menampilkan grafik historis
    st.subheader(f"Tren Historis: {barang_dipilih}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data_barang['Bulan'], data_barang['Quantity'], marker='o', linestyle='-', color='blue', label='Data Aktual (Jan 2024 - Ags 2026)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    st.pyplot(fig)

    # 4. Tombol Eksekusi Prediksi ARIMA
    st.markdown("---")
    if st.button("Jalankan Prediksi 12 Bulan Ke Depan"):
        with st.spinner('Algoritma ARIMA sedang menghitung proyeksi...'):
            ts_data = data_barang.set_index('Bulan')['Quantity']
            
            # Membangun Model ARIMA
            model = ARIMA(ts_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Melakukan peramalan 12 langkah (bulan) ke depan
            forecast = model_fit.forecast(steps=12)
            
            # Membuat index bulan untuk hasil prediksi
            forecast_index = pd.date_range(start=ts_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
            forecast.index = forecast_index

            # Menampilkan Grafik Gabungan
            st.subheader("Hasil Peramalan Kebutuhan Stok (Sep 2026 - Ags 2027)")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(ts_data.index, ts_data.values, marker='o', linestyle='-', color='blue', label='Data Aktual')
            ax2.plot(forecast.index, forecast.values, marker='o', linestyle='--', color='red', label='Hasil Prediksi')
            
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax2.legend()
            st.pyplot(fig2)

            # Menampilkan tabel hasil peramalan
            st.write("**Rincian Angka Proyeksi:**")
            df_forecast = pd.DataFrame({'Bulan': forecast.index.strftime('%B %Y'), 'Prediksi Kebutuhan (Quantity)': forecast.values.round()})
            st.dataframe(df_forecast, use_container_width=True)
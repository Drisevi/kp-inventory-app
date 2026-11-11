import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')

st.title("Sistem Demand Forecasting Persediaan")
st.write("PT Samudra Utama Narapati - Oleh: Sri Devi")

# 1. IMPLEMENTASI CACHE MEMORI (Solusi untuk mengatasi web lambat / lagging)
@st.cache_data
def load_and_clean_data(file):
    # Baca file Excel atau CSV
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    # 2. VALIDASI KOLOM (Skenario Uji Black-Box: Menolak file yang salah format)
    required_cols = ['Invoice Date', 'Item Description', 'Quantity']
    if not all(col in df.columns for col in required_cols):
        return None, "Gagal: Format data tidak sesuai. Pastikan file memiliki kolom 'Invoice Date', 'Item Description', dan 'Quantity'."
    
    # Pembersihan Data Lanjutan (Drop nilai kosong)
    df = df.dropna(subset=['Invoice Date', 'Item Description', 'Quantity'])
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df = df.dropna(subset=['Invoice Date', 'Quantity'])
    
    return df, "Data berhasil dibaca dan siap dianalisis!"

# Komponen Unggah File
uploaded_file = st.file_uploader("Unggah file Penjualan (Excel/CSV)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    df, message = load_and_clean_data(uploaded_file)
    
    if df is None:
        st.error(message) # Menampilkan pesan error jika kolom salah
    else:
        st.success(message)
        
        # Komponen Pilihan Produk (Dropdown)
        produk_list = df['Item Description'].unique()
        produk_pilihan = st.selectbox("Pilih Barang untuk Diproyeksikan:", produk_list)
        
        if produk_pilihan:
            # Filter data sesuai produk yang dipilih
            df_produk = df[df['Item Description'] == produk_pilihan]
            
            # 3. PENANGANAN BULAN KOSONG / RESAMPLING (Solusi ValueError ARIMA)
            # Dikelompokkan per awal bulan (Month Start) dan diisi 0 jika tidak ada penjualan
            ts_data = df_produk.set_index('Invoice Date')['Quantity'].resample('MS').sum().fillna(0)
            
            # Menampilkan Grafik Tren Historis
            st.subheader(f"Tren Historis: {produk_pilihan}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ts_data.index, ts_data.values, marker='o', label='Data Aktual', color='blue')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
            
            # Tombol Eksekusi Prediksi
            if st.button("Jalankan Prediksi 12 Bulan Ke Depan"):
                try:
                    # 4. PEMODELAN ARIMA(1,1,1)
                    model = ARIMA(ts_data, order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    # Melakukan peramalan 12 langkah (bulan) ke depan
                    forecast = model_fit.forecast(steps=12)
                    
                    # Menampilkan Grafik Proyeksi (Gabungan)
                    st.subheader(f"Hasil Peramalan Stok: {produk_pilihan} (12 Bulan Kedepan)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.plot(ts_data.index, ts_data.values, marker='o', label='Data Aktual', color='blue')
                    ax2.plot(forecast.index, forecast.values, marker='o', linestyle='--', label='Hasil Prediksi', color='red')
                    ax2.grid(True, linestyle='--', alpha=0.6)
                    ax2.legend()
                    st.pyplot(fig2)
                    
                    # Menampilkan Tabel Rincian Angka Prediksi
                    st.write("Rincian Angka Proyeksi:")
                    forecast_df = forecast.reset_index()
                    forecast_df.columns = ['Bulan', 'Prediksi Kebutuhan (Quantity)']
                    forecast_df['Bulan'] = forecast_df['Bulan'].dt.strftime('%B %Y')
                    # Membulatkan angka kuantitas agar tidak ada nilai desimal (karena barang fisik)
                    forecast_df['Prediksi Kebutuhan (Quantity)'] = forecast_df['Prediksi Kebutuhan (Quantity)'].round().astype(int)
                    
                    st.dataframe(forecast_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses ARIMA pada produk ini: {e}")

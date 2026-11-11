import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Dashboard Peramalan Penjualan - PT SUN",
    layout="wide"
)

# Judul Dashboard (Selaras dengan Judul Laporan Revisi)
st.title("Dashboard Peramalan Penjualan Berbasis ARIMA")
st.caption("PT Samudra Utama Narapati | Inventory Data Analyst: Sri Devi (2022071086)")
st.markdown("---")


# 1. FUNGSI PEMBACAAN & PEMBERSIHAN DATA DENGAN CACHE MEMORI
@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        return None, f"Gagal membaca berkas: {e}"

    # Rapikan spasi atau karakter baris baru pada nama kolom jika ada
    df.columns = [str(col).replace("\n", " ").strip() for col in df.columns]

    # Validasi keberadaan kolom wajib (Sesuai Skenario Uji Tabel 3.3)
    required_cols = ["Invoice Date", "Item Description", "Quantity"]
    if not all(col in df.columns for col in required_cols):
        return (
            None,
            "Format berkas tidak sesuai! Pastikan berkas memiliki kolom: "
            "'Invoice Date', 'Item Description', dan 'Quantity'.",
        )

    # Hanya ambil 3 kolom utama (Keamanan Data: Abaikan Customer Name / Harga)
    df = df[required_cols].copy()

    # Konversi tipe data dan hapus baris tidak valid
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Invoice Date", "Item Description", "Quantity"])

    # Urutkan berdasarkan tanggal
    df = df.sort_values("Invoice Date").reset_index(drop=True)

    return df, (
        f"Data berhasil dimuat: {len(df):,} baris transaksi valid "
        f"dari {df['Item Description'].nunique()} varian produk."
    )


# 2. KOMPONEN UNGGAH BERKAS
uploaded_file = st.file_uploader(
    "Unggah Berkas Data Penjualan Bersih (.xlsx / .xls / .csv):",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:
    df, status_msg = load_and_clean_data(uploaded_file)

    if df is None:
        st.error(status_msg)
    else:
        st.success(status_msg)

        # Menentukan rentang bulan utuh dari keseluruhan dataset (Jan 2024 - Ags 2026)
        min_date = df["Invoice Date"].min().replace(day=1)
        max_date = df["Invoice Date"].max().replace(day=1)
        full_month_index = pd.date_range(start=min_date, end=max_date, freq="MS")

        # 3. KOMPONEN PEMILIHAN PRODUK (DROPDOWN)
        produk_list = sorted(df["Item Description"].unique())
        default_idx = (
            produk_list.index("NARA UP KEMASAN 1 LITER")
            if "NARA UP KEMASAN 1 LITER" in produk_list
            else 0
        )
        produk_pilihan = st.selectbox(
            "Pilih Varian Produk untuk Dianalisis:",
            options=produk_list,
            index=default_idx
        )

        if produk_pilihan:
            # Filter data per produk
            df_produk = df[df["Item Description"] == produk_pilihan]

            # Agregasi bulanan (Month Start) & pengisian nilai 0 pada bulan tanpa transaksi
            ts_raw = (
                df_produk.set_index("Invoice Date")["Quantity"]
                .resample("MS")
                .sum()
            )
            ts_data = ts_raw.reindex(full_month_index, fill_value=0).fillna(0)
            ts_data.index.name = "Invoice Date"

            # Ringkasan metrik singkat produk terpilih
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Periode Pengamatan", f"{len(ts_data)} Bulan")
            col2.metric("Rata-rata Penjualan / Bulan", f"{ts_data.mean():,.0f} Unit")
            col3.metric("Penjualan Tertinggi / Bulan", f"{ts_data.max():,.0f} Unit")

            # 4. VISUALISASI GRAFIK TREN HISTORIS (Untuk Gambar 3.1)
            st.subheader(f"Grafik Tren Penjualan Historis: {produk_pilihan}")
            fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=300)
            ax1.plot(
                ts_data.index,
                ts_data.values,
                marker="o",
                color="#1f77b4",
                linewidth=1.8,
                label="Penjualan Aktual"
            )
            ax1.set_title(f"Tren Penjualan Bulanan Historis ({produk_pilihan})", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Periode Bulan", fontsize=10)
            ax1.set_ylabel("Kuantitas Penjualan (Unit)", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(loc="upper right")
            fig1.tight_layout()
            st.pyplot(fig1)

            st.markdown("---")

            # 5. EKSEKUSI PERAMALAN ARIMA(1,1,1) (Untuk Gambar 3.2)
            if st.button("Jalankan Peramalan Penjualan 12 Bulan ke Depan", type="primary"):
                try:
                    model = ARIMA(ts_data, order=(1, 1, 1))
                    model_fit = model.fit()

                    # Proyeksi 12 bulan ke depan (Sep 2026 - Ags 2027)
                    forecast = model_fit.forecast(steps=12)
                    forecast = forecast.clip(lower=0)  # Hindari nilai prediksi negatif

                    st.subheader(
                        f"Hasil Proyeksi Kuantitas Penjualan 12 Bulan ke Depan: {produk_pilihan}"
                    )

                    # Grafik Gabungan Aktual vs Proyeksi
                    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=300)
                    ax2.plot(
                        ts_data.index,
                        ts_data.values,
                        marker="o",
                        color="#1f77b4",
                        linewidth=1.8,
                        label="Data Aktual (Jan 2024 – Ags 2026)"
                    )
                    ax2.plot(
                        forecast.index,
                        forecast.values,
                        marker="s",
                        linestyle="--",
                        color="#d62728",
                        linewidth=1.8,
                        label="Proyeksi ARIMA(1,1,1) (Sep 2026 – Ags 2027)"
                    )
                    ax2.set_title(
                        f"Perbandingan Penjualan Aktual dan Proyeksi ARIMA(1,1,1) - {produk_pilihan}",
                        fontsize=11,
                        fontweight="bold"
                    )
                    ax2.set_xlabel("Periode Bulan", fontsize=10)
                    ax2.set_ylabel("Kuantitas Penjualan (Unit)", fontsize=10)
                    ax2.grid(True, linestyle="--", alpha=0.6)
                    ax2.legend(loc="upper right")
                    fig2.tight_layout()
                    st.pyplot(fig2)

                    # Tabel Rincian Angka Proyeksi 12 Bulan
                    st.write("**Tabel Rincian Estimasi Kuantitas Penjualan Bulanan:**")
                    forecast_df = pd.DataFrame({
                        "No.": range(1, 13),
                        "Periode Bulan": forecast.index.strftime("%B %Y"),
                        "Estimasi Penjualan (Unit)": forecast.round().astype(int).values
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Terjadi kendala perhitungan model ARIMA pada produk ini: {e}")

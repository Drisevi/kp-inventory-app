import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# 1. Gunakan layout="centered" agar lebar halaman proporsional saat di-screenshot
st.set_page_config(
    page_title="Dashboard Peramalan Penjualan - PT SUN",
    layout="centered"
)

# Mengurangi jarak kosong di bagian atas halaman agar muat dalam 1 layar
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dashboard Peramalan Penjualan Berbasis ARIMA")
st.caption("PT Samudra Utama Narapati | Inventory Data Analyst: Sri Devi (2022071086)")


@st.cache_data
def load_and_clean_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        return None, f"Gagal membaca berkas: {e}"

    df.columns = [str(col).replace("\n", " ").strip() for col in df.columns]

    required_cols = ["Invoice Date", "Item Description", "Quantity"]
    if not all(col in df.columns for col in required_cols):
        return (
            None,
            "Format berkas tidak sesuai! Pastikan berkas memiliki kolom: "
            "'Invoice Date', 'Item Description', dan 'Quantity'.",
        )

    df = df[required_cols].copy()
    df["Invoice Date"] = pd.to_datetime(df["Invoice Date"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Invoice Date", "Item Description", "Quantity"])
    df = df.sort_values("Invoice Date").reset_index(drop=True)

    return df, (
        f"Data berhasil dimuat: {len(df):,} baris transaksi valid "
        f"dari {df['Item Description'].nunique()} varian produk."
    )


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

        min_date = df["Invoice Date"].min().replace(day=1)
        max_date = df["Invoice Date"].max().replace(day=1)
        full_month_index = pd.date_range(start=min_date, end=max_date, freq="MS")

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
            df_produk = df[df["Item Description"] == produk_pilihan]

            ts_raw = (
                df_produk.set_index("Invoice Date")["Quantity"]
                .resample("MS")
                .sum()
            )
            ts_data = ts_raw.reindex(full_month_index, fill_value=0).fillna(0)
            ts_data.index.name = "Invoice Date"

            col1, col2, col3 = st.columns(3)
            col1.metric("Periode Pengamatan", f"{len(ts_data)} Bulan")
            col2.metric("Rata-rata / Bulan", f"{ts_data.mean():,.0f} Unit")
            col3.metric("Tertinggi / Bulan", f"{ts_data.max():,.0f} Unit")

            # 2. Ukuran grafik diperkecil (figsize=(7, 2.7)) agar pas dengan teks UI
            st.subheader(f"Grafik Tren Penjualan Historis: {produk_pilihan}")
            fig1, ax1 = plt.subplots(figsize=(7, 2.7), dpi=150)
            ax1.plot(
                ts_data.index,
                ts_data.values,
                marker="o",
                markersize=4,
                color="#1f77b4",
                linewidth=1.5,
                label="Penjualan Aktual"
            )
            ax1.set_title(f"Tren Penjualan Bulanan Historis ({produk_pilihan})", fontsize=9.5, fontweight="bold")
            ax1.set_xlabel("Periode Bulan", fontsize=8.5)
            ax1.set_ylabel("Kuantitas Penjualan (Unit)", fontsize=8.5)
            ax1.tick_params(axis="both", labelsize=8)
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(loc="upper right", fontsize=8)
            fig1.tight_layout()
            st.pyplot(fig1, use_container_width=False)

            if st.button("Jalankan Peramalan Penjualan 12 Bulan ke Depan", type="primary"):
                try:
                    model = ARIMA(ts_data, order=(1, 1, 1))
                    model_fit = model.fit()

                    forecast = model_fit.forecast(steps=12)
                    forecast = forecast.clip(lower=0)

                    st.subheader(f"Hasil Proyeksi Penjualan 12 Bulan: {produk_pilihan}")

                    # Grafik Gabungan Aktual vs Proyeksi dengan ukuran proporsional
                    fig2, ax2 = plt.subplots(figsize=(7, 2.7), dpi=150)
                    ax2.plot(
                        ts_data.index,
                        ts_data.values,
                        marker="o",
                        markersize=4,
                        color="#1f77b4",
                        linewidth=1.5,
                        label="Data Aktual (Jan 2024 – Ags 2026)"
                    )
                    ax2.plot(
                        forecast.index,
                        forecast.values,
                        marker="s",
                        markersize=4,
                        linestyle="--",
                        color="#d62728",
                        linewidth=1.5,
                        label="Proyeksi ARIMA(1,1,1) (Sep 2026 – Ags 2027)"
                    )
                    ax2.set_title(
                        f"Perbandingan Penjualan Aktual & Proyeksi ARIMA(1,1,1) - {produk_pilihan}",
                        fontsize=9,
                        fontweight="bold"
                    )
                    ax2.set_xlabel("Periode Bulan", fontsize=8.5)
                    ax2.set_ylabel("Kuantitas Penjualan (Unit)", fontsize=8.5)
                    ax2.tick_params(axis="both", labelsize=8)
                    ax2.grid(True, linestyle="--", alpha=0.6)
                    ax2.legend(loc="upper right", fontsize=7.5)
                    fig2.tight_layout()
                    st.pyplot(fig2, use_container_width=False)

                    st.write("**Tabel Rincian Estimasi Kuantitas Penjualan Bulanan:**")
                    forecast_df = pd.DataFrame({
                        "No.": range(1, 13),
                        "Periode Bulan": forecast.index.strftime("%B %Y"),
                        "Estimasi Penjualan (Unit)": forecast.round().astype(int).values
                    })
                    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Terjadi kendala perhitungan model ARIMA pada produk ini: {e}")

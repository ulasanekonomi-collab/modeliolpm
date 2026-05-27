import streamlit as st
import pandas as pd
from modeliolpm_engine import process_model_iolpm

st.set_page_config(page_title="Model IOLPM Dashboard", layout="wide")
st.title("📊 DASHBOARD MODEL IO LINGKUNGAN-PASAR-MONETER")
st.write("Inovasi Perencanaan Makroekonomi | Pengembang: **Yuhka Sundaya**")
st.markdown("---")

# ==========================================
# SIDEBAR PANEL KONTROL INTERAKTIF
# ==========================================
st.sidebar.header("📁 DATA MASUKAN (INPUT)")
uploaded_file = st.sidebar.file_uploader("Unggah CSV Tabel IO 17 Sektor BPS", type=["csv"])

st.sidebar.header("🌲 REZIM 1: INDIKATOR LINGKUNGAN")
theta_air = st.sidebar.slider("Ketersediaan Air Bersih Wilayah", 0.1, 1.0, 1.0, 0.05)
theta_lahan = st.sidebar.slider("Rasio Tutupan Lahan Hijau (RTH)", 0.1, 1.0, 1.0, 0.05)
aqi_input = st.sidebar.slider("Indeks Kualitas Udara Real-time (AQI)", 20, 300, 50, 10)
theta_udara_calc = 100.0 / aqi_input if aqi_input > 100 else 1.0

st.sidebar.header("🏪 REZIM 2: DENSITAS POPULASI PASAR")
status_harga = st.sidebar.selectbox("Pasar Primer (Stabilitas Harga Hulu)", ["Stabil", "Fluktuatif", "Spekulatif/Kacau"])
utilisasi_mfg = st.sidebar.slider("Pasar Sekunder (Utilisasi Mesin Pabrik) %", 20, 100, 70, 5)
kompetisi_jasa = st.sidebar.selectbox("Pasar Tersier (Kompetisi Sektor Jasa)", ["Rendah / Blue Ocean", "Padat / Kompetitif", "Perang Harga"])

# ==========================================
# PROSES SIMULASI KETIKA FILE DISEDIAKAN
# ==========================================
if uploaded_file is not None:
    Z_prime, L_prime, multipliers, error_mode, theta_nat_calc, macro_indicators, sektor_names = process_model_iolpm(
        uploaded_file, theta_air, theta_lahan, theta_udara_calc, status_harga, utilisasi_mfg, kompetisi_jasa
    )
    
    # 1. Tampilkan Panel 5 Indikator Makro-Moneter
    st.header("🎰 Indikator Dampak Makro-Moneter Terintegrasi")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric(label="💰 PDB Nominal Sistem", value=f"Rp {macro_indicators['PDB Nominal Total']/1000:,.2f} M")
    with m2:
        st.metric(label="👥 Upah Pekerja (Rumah Tangga)", value=f"Rp {macro_indicators['Pendapatan Rumah Tangga']/1000:,.2f} M")
    with m3:
        st.metric(label="🏛️ Pajak Neto Pemerintah", value=f"Rp {macro_indicators['Penerimaan Pajak Neto']/1000:,.2f} M")
    with m4:
        st.metric(label="💨 Estimasi Volume Emisi", value=f"{macro_indicators['Indeks Emisi Polusi Efektif']:.2f} Ton")
    with m5:
        st.metric(label="🏦 Kebutuhan Uang Beredar (M2)", value=f"Rp {macro_indicators['Kebutuhan Uang Beredar (M2)']/1000:,.2f} M")
        
    st.markdown("---")
    
    # 2. Pembagian Kolom Hasil Analisis Struktural
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Nilai Multiplier Output Hasil Relaksasi Model")
        df_mult = pd.DataFrame({"Sektor Ekonomi BPS": sektor_names, "Daya Ungkit Multiplier": multipliers})
        st.dataframe(df_mult.style.format({"Daya Ungkit Multiplier": "{:.4f}"}))
        
    with col2:
        st.subheader("🧠 Catatan Analisis & Kritik Kebijakan (Gödelian Mode)")
        if error_mode:
            st.error("🚨 **Paralisis Struktural Terjadi!** Sistem kehilangan konsistensi logis untuk mengeksekusi modal. Penambahan uang beredar tidak akan menghasilkan pertumbuhan riil karena jepitan krisis ganda alam dan pasar.")
        else:
            if aqi_input > 150:
                st.warning("⚠️ **Stagflasi Biofisik Terdeteksi:** Nilai PDB terhitung naik semu karena lonjakan biaya transaksi akibat rusaknya ekosistem hulu. Hal ini menipu indikator moneter (M2 ikut membengkak palsu).")
            if kompetisi_jasa == "Perang Harga":
                st.info("ℹ️ **Erosi Sosial Tersier:** Kepadatan populasi usaha jasa memicu perang harga, menggerus kemampuan sektor tersier dalam mendistribusikan upah buruh secara optimal.")
else:
    st.info("👋 Selamat datang di MODEL IOLPM! Silakan unggah file CSV Tabel IO 17 Sektor BPS pada panel sidebar di sebelah kiri untuk melihat simulasi dampak makro-moneter.")

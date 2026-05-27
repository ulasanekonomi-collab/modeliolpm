import streamlit as st
import pandas as pd
from modeliolpm_engine import assemble_io_table_dynamic

st.set_page_config(page_title="Model IOLPM - Dynamic Layer", layout="wide")
st.title("📊 Layer Data Dinamis (MODEL IOLPM)")
st.write("Inovasi Perencanaan Makroekonomi | Pengembang: **Yuhka Sundaya**")
st.markdown("---")

# ==========================================
# SIDEBAR PANEL PENGUNGGAHAN MODULAR
# ==========================================
st.sidebar.header("📁 STRUKTUR DATA MASUKAN")
st.sidebar.write("Unggah file CSV terpisah yang sudah dibersihkan:")

file_z = st.sidebar.file_uploader("1. Matriks Transaksi Antara (Z)", type=["csv"])
file_p = st.sidebar.file_uploader("2. Matriks Input Primer (P)", type=["csv"])
file_y = st.sidebar.file_uploader("3. Matriks Permintaan Akhir (Y)", type=["csv"])

# ==========================================
# PROSES SIMULASI DAN AUDIT STRUKTUR
# ==========================================
if file_z and file_p and file_y:
    try:
        # Panggil fungsi perakit dinamis
        df_io_complete, Z, P, Y, sektor_names, X_base = assemble_io_table_dynamic(file_z, file_p, file_y)
        
        st.success("✅ **Konstruksi Data Dinamis Berhasil!** Nama sektor dan label indikator berhasil dibaca langsung dari file Anda.")
        
        # Metrik Deteksi Dimensi File Otomatis
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Sektor Terdeteksi", f"{len(sektor_names)} Sektor Ekonomi")
        c2.metric("Komponen Input Primer", f"{P.shape[0]} Indikator Baris")
        c3.metric("Komponen Permintaan Akhir", f"{Y.shape[1]} Indikator Kolom")
        
        st.markdown("---")
        
        # Tampilkan DataFrame Hasil Konsolidasi Semuanya
        st.header("📋 Hasil Konsolidasi Tabel Input-Output")
        st.write("Seluruh label baris dan kolom di bawah ini bersifat dinamis mengikuti struktur teks asli file CSV Anda:")
        
        st.dataframe(df_io_complete.style.format("{:,.2f}"))
        
    except Exception as e:
        st.error(f"🚨 **Kesalahan Pembacaan Elemen:** Pastikan format Kolom 1 adalah Kode dan Kolom 2 adalah Deskripsi Teks. Detail error: {e}")
else:
    st.info("👋 Silakan unggah ketiga komponen berkas matriks terpisah pada panel sidebar di sebelah kiri untuk menguji sistem pembacaan nama sektor secara dinamis.")

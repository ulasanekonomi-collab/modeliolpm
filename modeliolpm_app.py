import streamlit as st
import pandas as pd
from modeliolpm_engine import process_pure_transaction_matrix

st.set_page_config(page_title="IOLPM - Layer 1 Matriks Transaksi", layout="wide")
st.title("📊 Layer 1: Konstruksi Matriks Transaksi Antara")
st.write("Pengembang: **Kang Yuhka** | Riset Pemodelan Ekonomi Makro Terintegrasi")
st.markdown("---")

st.sidebar.header("📁 INPUT MATRIKS UTAMA")
st.sidebar.write("Tahap 1: Validasi Matriks Transaksi Antara (Z)")
file_z = st.sidebar.file_uploader("Unggah Berkas Transaksi Antara (CSV)", type=["csv"])

if file_z:
    try:
        # Panggil fungsi pemrosesan tahap 1
        df_z_final, Z, nama_sektor = process_pure_transaction_matrix(file_z)
        
        st.success("✅ **Matriks Transaksi Berhasil Dimuat!** Sistem berhasil membaca matriks persegi dan memadukan kode dengan deskripsi sektor.")
        
        # Tampilkan Informasi Diagnostik Ringkas
        c1, c2 = st.columns(2)
        c2.metric("Ukuran Array Data Murni (Z)", f"{Z.shape[0]} Baris x {Z.shape[1]} Kolom")
        c1.metric("Sektor Ekonomi Terbaca", f"{len(nama_sektor)} Sektor Domestik")
        
        st.markdown("---")
        
        # Tampilkan Tabel Transaksi Utama dengan Pairing Kode-Deskripsi
        st.header("📋 Matriks Transaksi Antara Ter-Pairing & Nilai Agregat")
        st.write("Tabel di bawah menampilkan transaksi antar sektor ekonomi lengkap dengan kalkulasi total input (vertikal) dan total output (horizontal):")
        
        # Gunakan format angka standar akuntansi dengan pemisah ribuan koma dan dua desimal
        st.dataframe(df_z_final.style.format("{:,.2f}"))
        
        # Berikan fitur analisis tambahan berupa ringkasan eksekutif top sektor
        st.markdown("---")
        st.subheader("💡 Ringkasan Analisis Struktur Sektoral (Transaksi Antara)")
        
        df_ringkasan = pd.DataFrame({
            "Nama Sektor": nama_sektor,
            "Total Output Antara (Row Sum)": Z.sum(axis=1),
            "Total Input Antara (Col Sum)": Z.sum(axis=0)
        }).set_index("Nama Sektor")
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("**3 Sektor Pemasok (Output Antara) Terbesar:**")
            st.dataframe(df_ringkasan.sort_values(by="Total Output Antara (Row Sum)", ascending=False).head(3).style.format("{:,.2f}"))
            
        with col_right:
            st.write("**3 Sektor Konsumen (Input Antara) Terbesar:**")
            st.dataframe(df_ringkasan.sort_values(by="Total Input Antara (Col Sum)", ascending=False).head(3).style.format("{:,.2f}"))
            
    except Exception as e:
        st.error(f"🚨 **Gagal Memproses Berkas:** Terjadi kendala saat membaca struktur kolom atau baris sektor. Pesan kesalahan: {e}")
else:
    st.info("👋 Silakan unggah file CSV Matriks Transaksi Antara Anda melalui panel di sebelah kiri untuk melihat rekonstruksi pairing data dan kalkulasi agregat sektoral awal.")

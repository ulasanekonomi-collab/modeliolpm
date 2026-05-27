import streamlit as st
from modeliolpm_engine import process_isolated_matrix

st.title("📊 Isolasi Matriks Transaksi 17x17")

file_z = st.file_uploader("Upload Matriks Transaksi Antara", type=["csv"])

if file_z:
    try:
        df_display, Z, sektor_names = process_isolated_matrix(file_z)
        st.write("Matriks berhasil dimuat:")
        st.dataframe(df_display)
        
        st.write(f"Dimensi terdeteksi: {Z.shape}")
    except Exception as e:
        st.error(f"Error: {e}")

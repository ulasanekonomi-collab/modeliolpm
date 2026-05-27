import numpy as np
import pandas as pd

def process_and_clean_csv(file_wrapper):
    """
    Membaca file uploader Streamlit dan mendeteksi pembatas (delimiter) otomatis.
    """
    try:
        df = pd.read_csv(file_wrapper, sep=';')
    except:
        df = pd.read_csv(file_wrapper, sep=',')
        
    df.columns = [str(col).strip() for col in df.columns]
    return df

def clean_cell_value(val):
    """
    Mengubah sel data menjadi float murni dan menghapus spasi ribuan (siluman).
    """
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace(' ', '')
    if s == '-' or s == '' or s == '.' or s == ',':
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def process_pure_transaction_matrix(file_transaksi):
    """
    Modul Tahap 1: Hanya memproses Matriks Transaksi Antara (Z) 17x17
    dan memasangkan kode kolom dengan deskripsi nama sektor asli.
    """
    df_raw = process_and_clean_csv(file_transaksi)
    
    # Identifikasi Kolom secara dinamis (Kolom 0 = Kode, Kolom 1 = Deskripsi)
    kolom_kode = df_raw.columns[0]
    kolom_nama = df_raw.columns[1]
    
    # Ekstraksi nama deskripsi sektor (17 Sektor pertama)
    sektor_names = df_raw.iloc[0:17, 1].astype(str).str.strip().tolist()
    
    # Ekstraksi matriks angka transaksi murni (17 baris x 17 kolom data angka mulai kolom ke-3)
    Z = df_raw.iloc[0:17, 2:19].applymap(clean_cell_value).values.astype(float)
    
    # --- KALKULASI AGREGAT DASAR ---
    # 1. Total Output Antara (Jumlah Baris Sektor secara Horizontal)
    total_output_antara = Z.sum(axis=1)
    
    # 2. Total Input Antara (Jumlah Kolom Sektor secara Vertikal)
    total_input_antara = Z.sum(axis=0)
    
    # --- PEMBUATAN TABEL PAIRING UNTUK VISUALISASI ---
    # Buat nama kolom baru gabungan: "1. Pertanian", "2. Pertambangan", dst.
    kolom_pairing = [f"{i+1}. {sektor_names[i]}" for i in range(17)]
    
    # Masukkan matriks Z ke DataFrame dengan indeks dan header yang sudah dipasangkan
    df_z_display = pd.DataFrame(Z, index=kolom_pairing, columns=kolom_pairing)
    
    # Tambahkan Kolom Tambahan: Total Output Antara (Horizontal)
    df_z_display["TOTAL OUTPUT ANTARA (Row Sum)"] = total_output_antara
    
    # Tambahkan Baris Tambahan di paling bawah: Total Input Antara (Vertical Sum)
    baris_input_total = list(total_input_antara) + [total_input_antara.sum()]
    df_z_display.loc["TOTAL INPUT ANTARA (Col Sum)"] = baris_input_total
    
    return df_z_display, Z, sektor_names

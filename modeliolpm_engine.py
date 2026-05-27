import numpy as np
import pandas as pd

def clean_val(x):
    """
    Fungsi untuk membersihkan angka dari spasi, tanda koma, dan karakter non-numerik.
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(' ', '').replace(',', '.')
    if s == '-' or s == '' or s == '.':
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def process_pure_transaction_matrix(file_path):
    """
    Engine V13: Menggunakan pembacaan berbasis posisi kolom (indeks)
    untuk menghindari KeyError.
    """
    try:
        # Membaca berkas dengan pembatas titik koma
        df = pd.read_csv(file_path, sep=';')
    except:
        df = pd.read_csv(file_path, sep=',')
    
    # 1. Pastikan nama kolom bersih dari spasi di awal/akhir
    df.columns = df.columns.str.strip()
    
    # 2. Ambil nama sektor dari kolom kedua (indeks 1) secara posisional
    # Kita tidak lagi menggunakan nama kolom seperti df['Deksripsi']
    sektor_names = df.iloc[0:17, 1].astype(str).str.strip().tolist()
    
    # 3. Ambil data matriks transaksi (indeks kolom 2 sampai 18)
    # Ini adalah area data angka 17 sektor
    Z_raw = df.iloc[0:17, 2:19]
    
    # Konversi ke float murni
    Z_numeric = Z_raw.applymap(clean_val)
    Z_val = Z_numeric.values.astype(float)
    
    # 4. Kalkulasi Agregat
    df_result = pd.DataFrame(Z_val, index=sektor_names, columns=sektor_names)
    df_result['TOTAL OUTPUT'] = Z_val.sum(axis=1)
    df_result.loc['TOTAL INPUT'] = list(Z_val.sum(axis=0)) + [Z_val.sum()]
    
    return df_result, Z_val, sektor_names

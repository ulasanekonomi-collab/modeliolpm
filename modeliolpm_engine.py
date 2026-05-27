import numpy as np
import pandas as pd

def process_and_clean_csv(file_wrapper):
    """
    Membaca file uploader Streamlit secara dinamis dan mendeteksi pembatas (delimiter).
    """
    try:
        df = pd.read_csv(file_wrapper, sep=';')
    except:
        df = pd.read_csv(file_wrapper, sep=',')
        
    df.columns = [str(col).strip() for col in df.columns]
    return df

def clean_cell_value(val):
    """
    Mengubah sel data menjadi float murni dan menghapus spasi pemisah ribuan.
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

def assemble_io_table_dynamic(file_transaksi, file_primer, file_akhir):
    """
    Engine Komputasi Model IOLPM V11 - Adaptasi Struktur Modular Permintaan Akhir Horizontal
    """
    # 1. Membaca Data Mentah Berkas CSV
    df_z_raw = process_and_clean_csv(file_transaksi)
    df_p_raw = process_and_clean_csv(file_primer)
    df_y_raw = process_and_clean_csv(file_akhir)

    # 2. Ekstraksi Label Baris Sektor dari Matriks Transaksi Antara (Z)
    # Kolom indeks 1 dipastikan merupakan kolom deskripsi nama 17 sektor
    kolom_nama_z = df_z_raw.columns[1]
    sektor_names = df_z_raw[kolom_nama_z].astype(str).str.strip().tolist()
    
    # Ekstraksi Label Baris Komponen Input Primer (P)
    kolom_nama_p = df_p_raw.columns[1]
    primer_names = df_p_raw[kolom_nama_p].astype(str).str.strip().tolist()
    
    # Ekstraksi Label Header Kolom Permintaan Akhir (Y) secara langsung dari horizontal
    # Mengambil nama kolom mulai dari kolom ke-3 (indeks 2) sampai akhir
    akhir_headers = df_y_raw.columns[2:].tolist()

    # 3. Ekstraksi dan Pembersihan Nilai Numerik ke Array Numpy
    Z = df_z_raw.iloc[0:17, 2:19].applymap(clean_cell_value).values.astype(float)
    P = df_p_raw.iloc[:, 2:].applymap(clean_cell_value).values.astype(float)
    Y = df_y_raw.iloc[0:17, 2:].applymap(clean_cell_value).values.astype(float)

    # 4. Kalkulasi Operasi Penjumlahan Agregat Sektoral
    total_permintaan_antara = Z.sum(axis=1, keepdims=True)
    total_permintaan_akhir = Y.sum(axis=1, keepdims=True)
    total_output = total_permintaan_antara + total_permintaan_akhir

    # 5. Penggabungan Blok Horizontal Atas [ Z | Total Antara | Y | Total Akhir | Total Output ]
    blok_atas = np.hstack((Z, total_permintaan_antara, Y, total_permintaan_akhir, total_output))
    
    # Penyusunan nama header kolom untuk DataFrame Konsolidasi
    sektor_headers = [f"Sektor {i+1}" for i in range(Z.shape[1])]
    headers_lengkap = sektor_headers + ["Total Permintaan Antara"] + akhir_headers + ["Total Permintaan Akhir", "Total Output / Input"]
    
    # Proteksi dimensi kolom DataFrame
    if blok_atas.shape[1] != len(headers_lengkap):
        headers_lengkap = [f"Kolom_{i}" for i in range(blok_atas.shape[1])]
    
    df_io_tumpuk = pd.DataFrame(blok_atas, index=sektor_names, columns=headers_lengkap)

    # 6. Penggabungan Blok Vertikal Bawah (Input Primer)
    padding_kanan = np.zeros((P.shape[0], blok_atas.shape[1] - Z.shape[1]))
    blok_bawah_primer = np.hstack((P, padding_kanan))
    
    for i, nama_primer in enumerate(primer_names):
        df_io_tumpuk.loc[nama_primer] = blok_bawah_primer[i]
        
    # Tambahkan baris akumulasi akhir total input
    df_io_tumpuk.loc["TOTAL OUTPUT / INPUT"] = total_output.flatten()

    X_base = total_output.flatten()
    X_base = np.where(X_base == 0, 1.0, X_base)

    return df_io_tumpuk, Z, P, Y, sektor_names, X_base

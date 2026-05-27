import numpy as np
import pandas as pd

def process_and_clean_csv(file_wrapper):
    """
    Membaca file uploader Streamlit secara dinamis, mendeteksi delimiter,
    dan memosisikan kolom secara konsisten.
    """
    try:
        df = pd.read_csv(file_wrapper, sep=';')
    except:
        df = pd.read_csv(file_wrapper, sep=',')
        
    df.columns = [str(col).strip() for col in df.columns]
    return df

def clean_cell_value(val):
    """
    Mengubah sel data menjadi float murni dan membersihkan spasi ribuan.
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
    Engine Komputasi Model IOLPM V10 - Sinkronisasi Header Dinamis dan Proteksi Mismatch Kolom
    """
    # 1. Membaca Data Mentah
    df_z_raw = process_and_clean_csv(file_transaksi)
    df_p_raw = process_and_clean_csv(file_primer)
    df_y_raw = process_and_clean_csv(file_akhir)

    # 2. Identifikasi Nama Kolom Indeks Awal (Kode dan Deskripsi)
    kolom_nama_z = df_z_raw.columns[1]
    kolom_nama_p = df_p_raw.columns[1]

    # 3. Ekstraksi Label Baris
    sektor_names = df_z_raw[kolom_nama_z].astype(str).str.strip().tolist()
    primer_names = df_p_raw[kolom_nama_p].astype(str).str.strip().tolist()
    
    # Ekstraksi Nama Kolom Permintaan Akhir secara dinamis (mengambil header dari kolom ke-3 dst)
    akhir_headers = df_y_raw.columns[2:].tolist()

    # 4. Ekstraksi dan Pembersihan Nilai Numerik ke Array Numpy
    Z = df_z_raw.iloc[:, 2:].applymap(clean_cell_value).values.astype(float)
    P = df_p_raw.iloc[:, 2:].applymap(clean_cell_value).values.astype(float)
    Y = df_y_raw.iloc[:, 2:].applymap(clean_cell_value).values.astype(float)

    # 5. Kalkulasi Operasi Penjumlahan Agregat Sektoral
    total_permintaan_antara = Z.sum(axis=1, keepdims=True)
    total_permintaan_akhir = Y.sum(axis=1, keepdims=True)
    total_output = total_permintaan_antara + total_permintaan_akhir

    # 6. Penggabungan Blok Horizontal Atas
    blok_atas = np.hstack((Z, total_permintaan_antara, Y, total_permintaan_akhir, total_output))
    
    # Pembuatan daftar header kolom baru yang dijamin sinkron secara matematis dengan panjang blok_atas
    sektor_headers = [f"Sektor {i+1}" for i in range(Z.shape[1])]
    headers_lengkap = sektor_headers + ["Total Permintaan Antara"] + akhir_headers + ["Total Permintaan Akhir", "Total Output / Input"]
    
    # Proteksi Overwrite Header jika panjang array tidak sesuai akibat kolom tambahan di file akhir
    if blok_atas.shape[1] != len(headers_lengkap):
        headers_lengkap = [f"Kolom_{i}" for i in range(blok_atas.shape[1])]
    
    df_io_tumpuk = pd.DataFrame(blok_atas, index=sektor_names, columns=headers_lengkap)

    # 7. Penggabungan Blok Vertikal Bawah (Input Primer)
    padding_kanan = np.zeros((P.shape[0], blok_atas.shape[1] - Z.shape[1]))
    blok_bawah_primer = np.hstack((P, padding_kanan))
    
    for i, nama_primer in enumerate(primer_names):
        df_io_tumpuk.loc[nama_primer] = blok_bawah_primer[i]
        
    # Tambahkan baris akumulasi akhir
    df_io_tumpuk.loc["TOTAL OUTPUT / INPUT"] = total_output.flatten()

    X_base = total_output.flatten()
    X_base = np.where(X_base == 0, 1.0, X_base)

    return df_io_tumpuk, Z, P, Y, sektor_names, X_base

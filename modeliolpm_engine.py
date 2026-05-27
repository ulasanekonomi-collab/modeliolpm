import numpy as np
import pandas as pd

def process_pure_transaction_matrix(file_path):
    """
    Versi 'Anti-Peluru': 
    Maca CSV, ngabersihan sadaya karakter non-angka, 
    sarta otomatis ngadeteksi matriks 17x17.
    """
    # 1. Maca file
    try:
        df = pd.read_csv(file_path, sep=';', header=0)
    except:
        df = pd.read_csv(file_path, sep=',', header=0)
    
    # 2. Ambil 17 baris kahiji, sarta kolom angka (kolom ke-3 nepi ka ka-19)
    # Gunakeun 'coerce' sangkan nu lain angka langsung robah jadi NaN (0)
    data_asli = df.iloc[0:17, 2:19]
    Z = data_asli.apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # 3. Ngaran Sektor (Kolom ka-2 / indeks 1)
    sektor_names = df.iloc[0:17, 1].astype(str).tolist()
    
    # 4. Kalkulasi Dasar
    df_result = pd.DataFrame(Z, index=sektor_names, columns=sektor_names)
    df_result['TOTAL OUTPUT'] = Z.sum(axis=1)
    df_result.loc['TOTAL INPUT'] = list(Z.sum(axis=0)) + [Z.sum()]
    
    return df_result, Z, sektor_names

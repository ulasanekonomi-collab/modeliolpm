import numpy as np
import pandas as pd

def process_isolated_matrix(file_path):
    """
    Hanya membaca 17x17 data transaksi murni.
    """
    # Membaca CSV dengan deteksi delimiter
    try:
        df = pd.read_csv(file_path, sep=';')
    except:
        df = pd.read_csv(file_path, sep=',')
    
    # 1. Ambil Nama Sektor (Kolom ke-2, indeks 1)
    sektor_names = df.iloc[0:17, 1].astype(str).str.strip().tolist()
    
    # 2. Ambil Matriks Transaksi 17x17 murni 
    # Kita kunci mulai dari kolom indeks 2 sampai 18 (total 17 kolom)
    Z = df.iloc[0:17, 2:19].applymap(lambda x: float(str(x).replace(' ', '').replace(',', '.')) if pd.notnull(x) else 0.0).values
    
    # 3. Hitung Agregat
    total_output = Z.sum(axis=1)
    total_input = Z.sum(axis=0)
    
    # 4. Buat DataFrame untuk Streamlit
    df_result = pd.DataFrame(Z, index=sektor_names, columns=sektor_names)
    df_result['TOTAL OUTPUT'] = total_output
    df_result.loc['TOTAL INPUT'] = list(total_input) + [total_input.sum()]
    
    return df_result, Z, sektor_names

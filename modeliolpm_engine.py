import numpy as np
import pandas as pd

def process_pure_transaction_matrix(file_path):
    # Maca data tanpa ngandelkeun posisi kolom
    try:
        df = pd.read_csv(file_path, sep=';')
    except:
        df = pd.read_csv(file_path, sep=',')
    
    # 1. Pastikeun urang ngan ukur nyokot kolom nu ngaranna '1' nepi ka '17'
    # Ieu bakal otomatis ngabaékeun kolom '1800', '3011', jsb.
    kolom_target = [str(i) for i in range(1, 18)]
    
    # 2. Filter data (ngan ukur 17 baris kahiji jeung kolom 1-17)
    df_data = df.set_index(df.columns[1]) # Ngaran sektor jadi index
    Z = df_data[kolom_target].iloc[0:17]
    
    # Bersihkeun angka (hapus spasi, ganti koma jadi titik)
    Z_numeric = Z.applymap(lambda x: float(str(x).replace(' ', '').replace(',', '.')) if pd.notnull(x) else 0.0)
    
    # 3. Kalkulasi
    Z_val = Z_numeric.values
    df_result = Z_numeric.copy()
    df_result['TOTAL OUTPUT'] = Z_val.sum(axis=1)
    df_result.loc['TOTAL INPUT'] = list(Z_val.sum(axis=0)) + [Z_val.sum()]
    
    return df_result, Z_val, Z_numeric.index.tolist()

import numpy as np
import pandas as pd

def process_pure_transaction_matrix(file_path):
    # 1. Maca file
    try:
        df = pd.read_csv(file_path, sep=';')
    except:
        df = pd.read_csv(file_path, sep=',')
    
    # 2. Urang "intip" naon nami kolomna sacara otomatis
    # Ulah nulis 'Deskripsi' atawa 'Deksripsi' sacara manual, 
    # tapi cokot kolom kadua (indeks 1) naon waé ngaranna.
    kolom_deskripsi = df.columns[1]
    
    # 3. Filter kolom nu ngaranna angka '1' nepi ka '17'
    # Urang pastikeun ukur kolom éta nu dicokot
    kolom_target = [str(i) for i in range(1, 18)]
    
    # 4. Ambil data: baris 0-16 (17 sektor), kolom target
    Z_df = df.iloc[0:17][kolom_target]
    
    # 5. Bersihkeun data angka
    Z_numeric = Z_df.applymap(lambda x: float(str(x).replace(' ', '').replace(',', '.')) if pd.notnull(x) else 0.0)
    Z_val = Z_numeric.values
    
    # 6. Ambil nami sektor tina kolom deskripsi
    sektor_names = df.iloc[0:17][kolom_deskripsi].astype(str).tolist()
    
    # 7. Kalkulasi
    df_result = pd.DataFrame(Z_val, index=sektor_names, columns=sektor_names)
    df_result['TOTAL OUTPUT'] = Z_val.sum(axis=1)
    df_result.loc['TOTAL INPUT'] = list(Z_val.sum(axis=0)) + [Z_val.sum()]
    
    return df_result, Z_val, sektor_names

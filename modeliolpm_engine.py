import numpy as np
import pandas as pd

def process_model_iolpm(file_path, theta_air, theta_lahan, theta_udara, status_harga, utilisasi, kompetisi):
    """
    Engine Komputasi Model IOLPM V2 - Komputasi Dinamis Berbasis Kode BPS
    Anti-Lolos dari Error String/Pergeseran Baris Excel
    """
    # 1. Membaca Data Mentah CSV BPS
    df = pd.read_csv(file_path, skiprows=3, header=None)
    
    # Bersihkan kolom indeks 1 (Kode) menjadi string bersih agar mudah dicari
    df[1] = df[1].astype(str).str.strip()
    
    # --- PROSES EKSTRAKSI MATRIKS TRANSAKSI ANTARA (17 SEKTOR) ---
    # Sektor 1 sampai 17 selalu berada di 17 baris pertama setelah skiprows
    Z_base = df.iloc[1:18, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    
    # --- PROSES PENCARIAN DINAMIS BARIS SATELIT MAKRO ---
    
    # A. Cari Baris Total Input (Kode 2500 atau teks 'Total Input')
    row_X = df[df[1].str.contains('2500|Total Input', na=False, case=False)]
    if not row_X.empty:
        X_row = row_X.iloc[0, 3:20].apply(pd.to_numeric, errors='coerce').fillna(1.0).values.astype(float)
    else:
        # Fallback jika kode tidak ketemu, ambil baris paling terakhir dari dataframe
        X_row = df.iloc[-1, 3:20].apply(pd.to_numeric, errors='coerce').fillna(1.0).values.astype(float)
    X_base = np.where(X_row == 0, 1.0, X_row)

    # B. Cari Baris Kompensasi Tenaga Kerja (Kode 2010)
    row_upah = df[df[1].str.contains('2010|Kompensasi', na=False, case=False)]
    if not row_upah.empty:
        upah_base = row_upah.iloc[0, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    else:
        upah_base = df.iloc[19, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        
    # C. Cari Baris Pajak Neto Kurang Subsidi (Kode 2100 atau 1900)
    row_pajak = df[df[1].str.contains('2100|1900|Pajak', na=False, case=False)]
    if not row_pajak.empty:
        pajak_base = row_pajak.iloc[0, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    else:
        pajak_base = df.iloc[22, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

    # Nama Label Sektor untuk Komponen Antarmuka
    sektor_names = ["Pertanian", "Pertambangan", "Manufaktur", "Listrik & Gas", "Air & Limbah", "Konstruksi", 
                    "Perdagangan", "Transportasi", "Kuliner & Akomodasi", "Infokom", "Keuangan", "Real Estate", 
                    "Jasa Perusahaan", "Pemerintahan", "Pendidikan", "Kesehatan", "Jasa Lainnya"]

    # 2. Hitung Parameter Gubahan Model IOLPM
    theta_nat = (theta_air * theta_lahan * theta_udara) ** (1/3)
    f_nat = 1.0 / theta_nat
    
    mapping_harga = {"Stabil": 1.0, "Fluktuatif": 0.75, "Spekulatif/Kacau": 0.50}
    theta_mkt_primer = mapping_harga.get(status_harga, 1.0)
    theta_mkt_sekunder = 1.0 - (utilisasi / 100.0) ** 2
    
    mapping_jasa = {"Rendah / Blue Ocean": 1.0, "Padat / Kompetitif": 0.70, "Perang Harga": 0.35}
    theta_mkt_tersier = mapping_jasa.get(kompetisi, 1.0)

    # 3. Penyusunan Matriks Operator Relaksasi Omega (17x17)
    Omega = np.ones((17, 17))
    Omega[0:2, :] *= f_nat * theta_mkt_primer # Baris Sektor Primer
    Omega[:, 2:6] *= theta_mkt_sekunder       # Kolom Sektor Sekunder
    Omega[:, 6:17] *= theta_mkt_tersier       # Kolom Sektor Tersier
    
    # Jalankan proses Asimilasi Transaksi
    Z_prime = Z_base * Omega
    
    # 4. Hitung Koefisien Teknis A' & Invers Leontief L'
    A_prime = np.zeros((17, 17))
    for j in range(17):
        A_prime[:, j] = Z_prime[:, j] / X_base[j]
        
    I = np.eye(17)
    try:
        L_prime = np.linalg.inv(I - A_prime)
        error_mode = np.any(L_prime < 0) or np.any(np.isnan(L_prime))
    except np.linalg.LinAlgError:
        L_prime = np.zeros((17, 17))
        error_mode = True
        
    multipliers = L_prime.sum(axis=0) if not error_mode else np.zeros(17)

    # 5. Penghitungan Variabel Makroekonomi & Moneter (Fisher MV = PQ)
    macro_indicators = {}
    if not error_mode:
        distorsi_faktor = np.array([1.0, 1.0] + [theta_mkt_sekunder]*4 + [theta_mkt_tersier]*11)
        upah_distorted = upah_base * distorsi_faktor
        pajak_distorted = pajak_base * distorsi_faktor
        
        polusi_total = np.sum(Z_prime[0:2, :]) * (2.0 - theta_mkt_sekunder) * 0.000001
        total_Y = 16980285.0
        
        PQ = np.sum(Z_prime) + total_Y
        V_assumed = 2.5
        uang_beredar = PQ / V_assumed
        
        macro_indicators = {
            "PDB Nominal Total": np.sum(X_base * f_nat),
            "Pendapatan Rumah Tangga": np.sum(upah_distorted),
            "Penerimaan Pajak Neto": np.sum(pajak_distorted),
            "Indeks Emisi Polusi Efektif": polusi_total,
            "Kebutuhan Uang Beredar (M2)": uang_beredar
        }
    else:
        macro_indicators = {
            "PDB Nominal Total": 0.0, "Pendapatan Rumah Tangga": 0.0,
            "Penerimaan Pajak Neto": 0.0, "Indeks Emisi Polusi Efektif": 0.0,
            "Kebutuhan Uang Beredar (M2)": 0.0
        }

    return Z_prime, L_prime, multipliers, error_mode, theta_nat, macro_indicators, sektor_names

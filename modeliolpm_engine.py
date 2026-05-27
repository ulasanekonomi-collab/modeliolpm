import numpy as np
import pandas as pd

def process_model_iolpm(file_path, theta_air, theta_lahan, theta_udara, status_harga, utilisasi, kompetisi):
    # 1. Membaca Data Mentah CSV BPS (Gunakan baris header kode secara otomatis)
    df = pd.read_csv(file_path, skiprows=3, header=None)
    
    # Deteksi nama sektor dari baris 0 sampai 16, kolom indeks 2
    sektor_names = df.iloc[0:17, 2].tolist()
    
    # Ambil kolom indeks ke-3 sampai ke-19 (Karakteristik 17 Kolom Sektor Utama)
    # Kita pastikan konversi paksa ke numerik, jika ada sel teks kosong/aneh diubah jadi NaN lalu diisi 0
    Z_base = df.iloc[0:17, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    
    # Ambil baris Total Input (Indeks 23) khusus untuk kolom 3 sampai 19
    X_row = df.iloc[23, 3:20].apply(pd.to_numeric, errors='coerce').fillna(1.0).values.astype(float)
    X_base = np.where(X_row == 0, 1.0, X_row) # Proteksi pembagian nol
    
    # Ambil upah tenaga kerja (Indeks 18) khusus untuk kolom 3 sampai 19
    upah_base = df.iloc[18, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
        
    # Ambil pajak kurang subsidi (Indeks 21) khusus untuk kolom 3 sampai 19
    pajak_base = df.iloc[21, 3:20].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)

    # 2. Hitung Parameter Gubahan Model IOLPM
    theta_nat = (theta_air * theta_lahan * theta_udara) ** (1/3)
    f_nat = 1.0 / theta_nat
    
    mapping_harga = {"Stabil": 1.0, "Fluktuatif": 0.75, "Spekulatif/Kacau": 0.50}
    theta_mkt_primer = mapping_harga.get(status_harga, 1.0)
    theta_mkt_sekunder = 1.0 - (utilisasi / 100.0) ** 2
    
    mapping_jasa = {"Rendah / Blue Ocean": 1.0, "Padat / Competitive": 0.70, "Perang Harga": 0.35}
    theta_mkt_tersier = mapping_jasa.get(kompetisi, 1.0)

    # 3. Penyusunan Matriks Operator Relaksasi Omega (17x17)
    Omega = np.ones((17, 17))
    Omega[0:2, :] *= f_nat * theta_mkt_primer # Baris Primer
    Omega[:, 2:6] *= theta_mkt_sekunder       # Kolom Sekunder
    Omega[:, 6:17] *= theta_mkt_tersier       # Kolom Tersier
    
    # Asimilasi transaksi ter-relaksasi
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

    # 5. Penghitungan Variabel Makroekonomi & Moneter
    macro_indicators = {}
    if not error_mode:
        distorsi_faktor = np.array([1.0, 1.0] + [theta_mkt_sekunder]*4 + [theta_mkt_tersier]*11)
        upah_distorted = upah_base * distorsi_faktor
        pajak_distorted = pajak_base * distorsi_faktor
        
        polusi_total = np.sum(Z_prime[0:2, :]) * (2.0 - theta_mkt_sekunder) * 0.00001
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

import numpy as np
import pandas as pd

def clean_val(x):
    """
    Fungsi khusus untuk menghapus spasi di awal, akhir, serta spasi siluman
    di tengah angka sebagai pemisah ribuan bawaan dari file CSV BPS.
    """
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(' ', '')
    if s == '-' or s == '' or s == '.' or s == ',':
        return 0.0
    try:
        return float(s)
    except:
        return 0.0

def process_model_iolpm(file_path, theta_air, theta_lahan, theta_udara, status_harga, utilisasi, kompetisi):
    """
    Engine Komputasi Model IOLPM V5 - Koordinat Absolut Terkalibrasi 100%
    """
    # 1. Membaca Data Mentah CSV BPS (Menggunakan pemisah titik koma ';')
    try:
        df = pd.read_csv(file_path, sep=';', header=None)
    except:
        df = pd.read_csv(file_path, sep=',', header=None)
        
    # --- EKSTRAKSI ABSOLUT DATA MATRIKS DAN SATELIT MAKRO ---
    # Berdasarkan struktur fisik berkas CSV BPS 17 Sektor Domestik:
    
    # Matriks Transaksi Antara 17x17 (Baris indeks 5 sampai 21, Kolom indeks 3 sampai 19)
    Z_base = df.iloc[5:22, 3:20].applymap(clean_val).values.astype(float)
    
    # Baris Total Input / Output Domestik (X) berada tepat pada Baris Indeks 30
    X_base = df.iloc[30, 3:20].apply(clean_val).values.astype(float)
    X_base = np.where(X_base == 0, 1.0, X_base) # Proteksi pembagian nol

    # Baris Kompensasi Tenaga Kerja / Upah berada tepat pada Baris Indeks 23
    upah_base = df.iloc[23, 3:20].apply(clean_val).values.astype(float)
        
    # Baris Pajak Neto Kurang Subsidi atas Produk berada tepat pada Baris Indeks 22
    pajak_base = df.iloc[22, 3:20].apply(clean_val).values.astype(float)

    # Label Sektor Resmi untuk Keperluan Visualisasi Tabel Dasbor
    sektor_names = ["Pertanian", "Pertambangan", "Manufaktur", "Listrik & Gas", "Air & Limbah", "Konstruksi", 
                    "Perdagangan", "Transportasi", "Kuliner & Akomodasi", "Infokom", "Keuangan", "Real Estate", 
                    "Jasa Perusahaan", "Pemerintahan", "Pendidikan", "Kesehatan", "Jasa Lainnya"]

    # 2. Perhitungan Parameter Gubahan Model IOLPM
    theta_nat = (theta_air * theta_lahan * theta_udara) ** (1/3)
    f_nat = 1.0 / theta_nat
    
    mapping_harga = {"Stabil": 1.0, "Fluktuatif": 0.75, "Spekulatif/Kacau": 0.50}
    theta_mkt_primer = mapping_harga.get(status_harga, 1.0)
    theta_mkt_sekunder = 1.0 - (utilisasi / 100.0) ** 2
    
    mapping_jasa = {"Rendah / Blue Ocean": 1.0, "Padat / Kompetitif": 0.70, "Perang Harga": 0.35}
    theta_mkt_tersier = mapping_jasa.get(kompetisi, 1.0)

    # 3. Penyusunan Matriks Operator Relaksasi Omega (17x17)
    Omega = np.ones((17, 17))
    Omega[0:2, :] *= f_nat * theta_mkt_primer # Pembatasan Baris Sektor Primer
    Omega[:, 2:6] *= theta_mkt_sekunder       # Pembatasan Kolom Sektor Sekunder
    Omega[:, 6:17] *= theta_mkt_tersier       # Pembatasan Kolom Sektor Tersier
    
    # Asimilasi Transaksi Ter-Relaksasi Z'
    Z_prime = Z_base * Omega
    
    # 4. Perhitungan Koefisien Teknis A' & Invers Leontief L'
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

    # 5. Perhitungan Variabel Makroekonomi & Moneter (Pendekatan Fisher MV = PQ)
    macro_indicators = {}
    if not error_mode:
        distorsi_faktor = np.array([1.0, 1.0] + [theta_mkt_sekunder]*4 + [theta_mkt_tersier]*11)
        upah_distorted = upah_base * distorsi_faktor
        pajak_distorted = pajak_base * distorsi_faktor
        
        polusi_total = np.sum(Z_prime[0:2, :]) * (2.0 - theta_mkt_sekunder) * 0.000001
        total_Y = 16980285.0 # Estimasi Permintaan Akhir Agregat
        
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

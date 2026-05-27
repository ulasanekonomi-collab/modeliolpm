import numpy as np
import pandas as pd

def process_model_iolpm(file_path, theta_air, theta_lahan, theta_udara, status_harga, utilisasi, kompetisi):
    """
    Engine Komputasi Model IO Lingkungan-Pasar-Moneter (IOLPM)
    Berbasis Data Riil 17 Sektor BPS 2020
    """
    # 1. Membaca Data Mentah CSV BPS (Melewati 3 baris judul atas)
    df = pd.read_csv(file_path, skiprows=3, header=None)
    
    # Ekstrak nama-nama sektor dari kolom indeks 2 (Baris 0 sampai 16)
    sektor_names = df.iloc[0:17, 2].tolist()
    
    # Ambil matriks transaksi antara 17x17 (Baris 0-16, Kolom indeks 3 sampai 19)
    Z_base = df.iloc[0:17, 3:20].values.astype(float)
    
    # Ambil baris Total Input/Output Domestik (Sekarang pas di indeks 23)
    X_base = df.iloc[23, 3:20].values.astype(float)
    X_base[X_base == 0] = 1.0 
    
    # Ambil data kompensasi tenaga kerja (Baris Kode 2010 -> indeks 18)
    upah_base = df.iloc[18, 3:20].values.astype(float)
        
    # Ambil data pajak kurang subsidi (Baris indeks 21)
    pajak_base = df.iloc[21, 3:20].values.astype(float)

    # 2. Hitung Parameter Gubahan Model IOLPM
    # REZIM 1: Rata-rata geometris indikator alam
    theta_nat = (theta_air * theta_lahan * theta_udara) ** (1/3)
    f_nat = 1.0 / theta_nat
    
    # REZIM 2: Densitas Pasar Sektoral
    mapping_harga = {"Stabil": 1.0, "Fluktuatif": 0.75, "Spekulatif/Kacau": 0.50}
    theta_mkt_primer = mapping_harga.get(status_harga, 1.0)
    
    # Formula logistik pembatas kejenuhan manufaktur
    theta_mkt_sekunder = 1.0 - (utilisasi / 100.0) ** 2
    
    mapping_jasa = {"Rendah / Blue Ocean": 1.0, "Padat / Kompetitif": 0.70, "Perang Harga": 0.35}
    theta_mkt_tersier = mapping_jasa.get(kompetisi, 1.0)

    # 3. Penyusunan Matriks Operator Relaksasi Omega (17x17)
    Omega = np.ones((17, 17))
    
    # Baris 0-1 (Sektor Primer: Pertanian & Pertambangan) terkena dampak kelangkaan hulu
    Omega[0:2, :] *= f_nat * theta_mkt_primer
    
    # Kolom 2-5 (Sektor Sekunder: Manufaktur, Listrik, Air, Konstruksi) terkena rem jenuh
    Omega[:, 2:6] *= theta_mkt_sekunder
    
    # Kolom 6-16 (Sektor Tersier: Jasa-jasa) terkena erosi marjin kompetisi
    Omega[:, 6:17] *= theta_mkt_tersier
    
    # Asimilasi matriks transaksi baru Z'
    Z_prime = Z_base * Omega
    
    # 4. Hitung Koefisien Teknis A' & Invers Leontief L'
    A_prime = Z_prime / X_base
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
        # Upah Rumah Tangga terdistorsi oleh shock kapasitas pasar kolom masing-masing
        distorsi_faktor = np.array([1.0, 1.0] + [theta_mkt_sekunder]*4 + [theta_mkt_tersier]*11)
        upah_distorted = upah_base * distorsi_faktor
        pajak_distorted = pajak_base * distorsi_faktor
        
        # Indeks Polusi berbanding lurus dengan pembengkokan ekstraksi dan aktivitas pabrik
        polusi_total = np.sum(Z_prime[0:2, :]) * (2.0 - theta_mkt_sekunder) * 0.00001
        
        # Total Permintaan Akhir Nominal (diambil konstanta agregat total kolom 3090 BPS)
        total_Y = 16980285.0  # Skala dalam juta rupiah
        
        # Moneter: Total Transaksi Nominal PQ = Total Transaksi Antara + Total Permintaan Akhir
        PQ = np.sum(Z_prime) + total_Y
        V_assumed = 2.5 # Kecepatan perputaran uang jangkar
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

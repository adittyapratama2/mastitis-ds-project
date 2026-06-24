# src/data_fusion.py
# -*- coding: utf-8 -*-
"""
Master's Thesis Artifact - S2 Big Data Amikom Yogyakarta
Author: Adittya Pratama & Team
Description: RIGOROUS FINE-TUNED PIPELINE - Resolved Over-Separation & Semantic Mismatch.
             Injected Overlapping Biological Noise and Synchronized Milk Yield Distributions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def execute_final_fusion():
    print("=============================================================")
    print("=== EXECUTING FINE-TUNED STOCHASTIC DATA FUSION PIPELINE ===")
    print("=============================================================")
    
    # 1. Amankan Struktur Lokasi (Lock Absolut ke docs/)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    root_project_dir = os.path.dirname(current_script_dir)
    raw_dir = os.path.join(root_project_dir, 'docs')
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    base_data_path = os.path.join(raw_dir, 'cow_milk_mastitis_dataset.csv')
    
    if not os.path.exists(base_data_path):
        print(f"[CRITICAL ERROR] File CSV dasar tidak ditemukan di: {base_data_path}")
        return
        
    df_base = pd.read_csv(base_data_path)
    num_samples = len(df_base)
    
    # Ambil karakteristik statistik riil Milk_Yield dari CSV dasar untuk sinkronisasi skenario sintetis
    real_yield_mean_healthy = df_base[df_base['class1'] == 0]['Milk_Yield'].mean()
    real_yield_std_healthy = df_base[df_base['class1'] == 0]['Milk_Yield'].std()
    real_yield_mean_sick = df_base[df_base['class1'] == 1]['Milk_Yield'].mean()
    real_yield_std_sick = df_base[df_base['class1'] == 1]['Milk_Yield'].std()

    print(f"[+] Berhasil memuat {num_samples} sampel dataset dasar.")
    print(f"[+] Statistik Milk Yield Riil - Sehat: {real_yield_mean_healthy:.2f} Kg, Sakit: {real_yield_mean_sick:.2f} Kg")

    # Mengunci generator acak demi reproduksibilitas riset
    np.random.seed(42)

    # ==========================================
    # SKENARIO A: DATA EMPIRIS (REAL + TUNED REALISTIC WEARABLE NOISE)
    # ==========================================
    print("\n[+] Memproses Skenario Empiris (Fine-Tuned Realistic Overlap)...")
    
    # Kalibrasi termal (+3.5°C) dari suhu susu ke estimasi suhu internal inti tubuh
    calibrated_body_temp = df_base['Milk_Temperature'] + 3.5
    
    # FINE-TUNING REKOMENDASI: Memperkecil koefisien efek dan memperbesar standard deviasi (noise)
    # Perbedaan rata-rata HR kini hanya ~6 bpm dengan noise standar deviasi 6 (SNR Rendah, Realistis)
    heart_rate_emp = 70 + (calibrated_body_temp - 38.5) * 8 + np.random.normal(0, 6, num_samples)
    
    # Efek penurunan ruminasi diperkecil menjadi ~77 menit dengan noise 40 menit (Overlap tinggi)
    rumination_emp = 480 - (df_base['Milk_Conductivity'] - 4.8) * 35 + np.random.normal(0, 40, num_samples)
    
    # Kliping batas biologis normal-ekstrim sapi perah
    calibrated_body_temp = np.clip(calibrated_body_temp, 37.0, 41.5)
    heart_rate_emp = np.clip(heart_rate_emp, 50, 130)
    rumination_emp = np.clip(rumination_emp, 100, 650)

    df_empirical = pd.DataFrame({
        'body_temperature': calibrated_body_temp,
        'heart_rate': heart_rate_emp,
        'rumination_time': rumination_emp,
        'milk_yield': df_base['Milk_Yield'], # Karakteristik yield riil lapangan lapangan
        'milk_conductivity': df_base['Milk_Conductivity'],
        'mastitis_status': df_base['class1']
    })

    # Stratified Split untuk mengunci proporsi label kelas imbalanced
    train_emp, test_emp = train_test_split(
        df_empirical,
        test_size=0.2,
        stratify=df_empirical['mastitis_status'],
        random_state=42
    )

    # ==========================================
    # SKENARIO B: DATA SINTETIS (SYNCHRONIZED METRIC DISTRIBUTION)
    # ==========================================
    print("[+] Memproses Skenario Sintetis (Sinkronisasi Sebaran & Full-Class Drift)...")
    
    status_labels = df_base['class1'].values

    def generate_synthetic_with_realistic_drift(labels, drift_factor=1.0):
        size = len(labels)
        body_temp = np.zeros(size)
        h_rate = np.zeros(size)
        r_time = np.zeros(size)
        m_yield = np.zeros(size)
        m_cond = np.zeros(size)
        
        for i in range(size):
            if labels[i] == 0:  # Sapi Sehat
                body_temp[i] = np.random.normal(38.5, 0.2 * drift_factor)
                m_cond[i] = np.random.normal(4.6, 0.2 * drift_factor)
                # SINKRONISASI 1: Ambil sebaran mean/std asli lapangan agar konsisten dengan empiris
                m_yield[i] = np.random.normal(real_yield_mean_healthy, real_yield_std_healthy * drift_factor)
            else:  # Sapi Mastitis Subklinis
                body_temp[i] = np.random.normal(39.3, 0.4 * drift_factor)
                m_cond[i] = np.random.normal(6.8, 0.4 * drift_factor)
                # SINKRONISASI 2: Ambil sebaran mean/std asli lapangan agar konsisten dengan empiris
                m_yield[i] = np.random.normal(real_yield_mean_sick, real_yield_std_sick * drift_factor)
                
            # Hubungan Berantai Fisiologis Realistis (Menerapkan Tuning Koefisien & Noise yang Sama)
            h_rate[i] = 70 + (body_temp[i] - 38.5) * 8 + np.random.normal(0, 6 * drift_factor)
            r_time[i] = 480 - (m_cond[i] - 4.8) * 35 + np.random.normal(0, 40 * drift_factor)
                
        return pd.DataFrame({
            'body_temperature': np.clip(body_temp, 37.0, 41.5),
            'heart_rate': np.clip(h_rate, 50, 130),
            'rumination_time': np.clip(r_time, 100, 650),
            'milk_yield': m_yield,
            'milk_conductivity': m_cond,
            'mastitis_status': labels
        })

    # Train dikunci pada baseline (1.0), Test didorong ke lingkungan ekstrim (1.3)
    df_synthetic_train_all = generate_synthetic_with_realistic_drift(status_labels, drift_factor=1.0)
    df_synthetic_test_all = generate_synthetic_with_realistic_drift(status_labels, drift_factor=1.3)

    # Gunakan indeks yang sama dengan stratified split empiris agar adil secara statistik
    train_syn = df_synthetic_train_all.iloc[train_emp.index]
    test_syn = df_synthetic_test_all.iloc[test_emp.index]

    # ==========================================
    # EKSPORT ARTEFAK DATA KE FOLDER /data
    # ==========================================
    train_emp.to_csv(os.path.join(output_dir, 'train_empirical.csv'), index=False)
    test_emp.to_csv(os.path.join(output_dir, 'test_empirical.csv'), index=False)
    train_syn.to_csv(os.path.join(output_dir, 'train_synthetic.csv'), index=False)
    test_syn.to_csv(os.path.join(output_dir, 'test_synthetic.csv'), index=False)
    
    print("\n" + "="*70)
    print("✔ FINE-TUNING SUKSES: DISTRIBUSI REALISTIS DAN SINKRON TELAH DIEKSPOR")
    print("=======================================================================")

if __name__ == '__main__':
    execute_final_fusion()
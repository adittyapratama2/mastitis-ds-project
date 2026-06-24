# src/train_dual.py
# -*- coding: utf-8 -*-
"""
Master's Thesis Artifact - S2 Big Data Amikom Yogyakarta
Author: Adittya Pratama & Team
Description: Final Production Cross-Scenario Validation Matrix.
             Automated Artifacts: Cross-Validation Metrics (.csv), Feature Importance Logs (.csv),
             Model Size Audit, and Hardware-Level Edge Latency Benchmark (Mean & P95).
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os
import time
import numpy as np

def benchmark_latency(model, X_sample, n_iter=1000):
    """
    Mengukur waktu inferensi model pada sampel data secara masal (micro-benchmarking).
    Mengembalikan mean latency (ms) dan P95 latency (ms).
    """
    # 1. Warm-up Loop (Proteksi CPU Cold Start & Cache Loading)
    for _ in range(20):
        _ = model.predict_proba(X_sample)
    
    # 2. Iterative Performance Measurement via High-Resolution Timer
    latencies = []
    for _ in range(n_iter):
        start = time.perf_counter()
        _ = model.predict_proba(X_sample)
        latencies.append((time.perf_counter() - start) * 1000) # Konversi ke milidetik (ms)
    
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    return mean_lat, p95_lat

def run_production_dual_scenario_matrix():
    print("=============================================================")
    print("=== STARTING ADVANCED CROSS-SCENARIO VALIDATION PIPELINE ===")
    print("=============================================================")
    
    data_dir = 'data'
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Ingestion Dataset (Train & Test) Hasil Stratified Split
    datasets = {}
    scenarios = ['empirical', 'synthetic']
    
    for sc in scenarios:
        train_path = os.path.join(data_dir, f'train_{sc}.csv')
        test_path = os.path.join(data_dir, f'test_{sc}.csv')
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"[CRITICAL ERROR] Berkas data untuk skenario '{sc}' tidak lengkap di /data.")
            print("Jalankan 'python src/data_fusion.py' terlebih dahulu.")
            return
            
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        
        datasets[f'X_train_{sc}'] = df_train.drop(columns=['mastitis_status'])
        datasets[f'y_train_{sc}'] = df_train['mastitis_status']
        datasets[f'X_test_{sc}'] = df_test.drop(columns=['mastitis_status'])
        datasets[f'y_test_{sc}'] = df_test['mastitis_status']

    models = {}
    latency_records = {}
    
    # 2. TAHAP 1: Pelatihan Model, OOB Validation, Feature Importance, & Latency Benchmark
    print("\n>>> TAHAP 1: PELATIHAN MODEL, OOB VALIDATION & EKSTRAKSI PERFORMANCE AUDIT <<<")
    for sc in scenarios:
        print(f"\n[+] Training Random Forest untuk Skenario: {sc.upper()}")
        
        model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            class_weight='balanced', 
            oob_score=True,
            random_state=42, 
            n_jobs=-1
        )
        
        X_train = datasets[f'X_train_{sc}']
        y_train = datasets[f'y_train_{sc}']
        X_test = datasets[f'X_test_{sc}']
        
        model.fit(X_train, y_train)
        models[sc] = model
        
        print(f"    |-- Out-of-Bag (OOB) Generalization Score : {model.oob_score_:.4f}")
        print("    |-- Feature Importance (Bobot Kontribusi Variabel):")
        
        importances = model.feature_importances_
        feature_names = X_train.columns
        feature_imp_df = pd.DataFrame({'Fitur': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        for index, row in feature_imp_df.iterrows():
            print(f"    |   └─ {row['Fitur']:22s} : {row['Importance']:.4f}")
            
        # INTEGRASI BENCHMARK LATENSI: Ambil 1 baris sampel dari data uji independen
        X_sample = X_test.iloc[:1]
        mean_lat, p95_lat = benchmark_latency(model, X_sample)
        latency_records[sc] = {"mean_ms": mean_lat, "p95_ms": p95_lat}
        
        print(f"    |-- [BENCHMARK] Mean Inference Latency    : {mean_lat:.4f} ms (Target: < 50 ms)")
        print(f"    |-- [BENCHMARK] P95 Inference Latency     : {p95_lat:.4f} ms")
        
        # Simpan Feature Importance ke file CSV terpisah
        feat_imp_output_path = os.path.join(model_dir, f'feature_importance_{sc}.csv')
        feature_imp_df.to_csv(feat_imp_output_path, index=False)
        print(f"    |─ Artefak Feature Importance dieksport ke : {feat_imp_output_path}")
            
        # Serialisasi biner model pkl
        model_output_path = os.path.join(model_dir, f'mastitis_model_{sc}.pkl')
        joblib.dump(model, model_output_path)
        print(f"    |─ File biner model berhasil dieksport ke  : {model_output_path}")
        
        # Model Size Audit
        file_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
        print(f"    |─ [AUDIT] Ukuran Fisik File Model Biner  : {file_size_mb:.4f} MB (Target: < 10 MB)")

    # 3. TAHAP 2: Eksekusi Cross-Scenario Validation Matrix dengan Metrik Komprehensif
    print("\n" + "="*115)
    print(">>> TAHAP 2: MATRIKS VALIDASI SILANG KOMPREHENSIF DENGAN BENCHMARK HARDWARE (DATA LAPORAN) <<<")
    print("="*115)
    print(f"{'Train Set':12s} | {'Test Set':12s} | {'Accuracy':8s} | {'Precision':9s} | {'Recall':8s} | {'F1-Score':8s} | {'ROC-AUC':8s} | {'Mean Lat (ms)':13s} | {'P95 Lat (ms)':12s}")
    print("-" * 115)
    
    matrix_scenarios = [
        ('empirical', 'empirical'),
        ('synthetic', 'synthetic'),
        ('empirical', 'synthetic'),
        ('synthetic', 'empirical')
    ]
    
    cross_val_results = []
    confusion_matrices = {}
    
    for train_sc, test_sc in matrix_scenarios:
        model = models[train_sc]
        X_test = datasets[f'X_test_{test_sc}']
        y_test = datasets[f'y_test_{test_sc}']
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        # Ambil record latensi hardware dari basis latih model yang bersangkutan
        m_lat = latency_records[train_sc]["mean_ms"]
        p_lat = latency_records[train_sc]["p95_ms"]
        
        confusion_matrices[f"{train_sc}_to_{test_sc}"] = confusion_matrix(y_test, y_pred)
        
        print(f"{train_sc:12s} | {test_sc:12s} | {acc:8.4f} | {prec:9.4f} | {rec:8.4f} | {f1:8.4f} | {auc:8.4f} | {m_lat:13.4f} | {p_lat:12.4f}")
        
        # Simpan record lengkap termasuk data latensi fisik ke file CSV laporan
        cross_val_results.append({
            "train_set": train_sc,
            "test_set": test_sc,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "mean_latency_ms": round(m_lat, 4),
            "p95_latency_ms": round(p_lat, 4)
        })
    
    print("-" * 115)
    
    # Dump seluruh hasil matriks lengkap ke bentuk file CSV fisik
    report_output_path = os.path.join(model_dir, "cross_validation_report.csv")
    pd.DataFrame(cross_val_results).to_csv(report_output_path, index=False)
    print(f"✔ [ARTEFAK LENGKAP AMAN] Laporan Matriks Evaluasi Silang + Latency disimpan di: {report_output_path}")
    
    print("\n>>> TAHAP 3: DETAIL CONFUSION MATRIX UNTUK EVALUASI ERROR METRICS <<<")
    for key, matrix in confusion_matrices.items():
        print(f"\n[+] Confusion Matrix Skenario [{key.replace('_to_', ' -> ').upper()}]:")
        print(f"    TN: {matrix[0][0]:<4} | FP: {matrix[0][1]}")
        print(f"    FN: {matrix[1][0]:<4} | TP: {matrix[1][1]}")
        
    print("=============================================================")
    print("=== PIPELINE VALIDASI SILANG SELESAI DAN SUKSES ===")
    print("=============================================================")

if __name__ == '__main__':
    run_production_dual_scenario_matrix()
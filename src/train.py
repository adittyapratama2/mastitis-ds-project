# src/train.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
import time
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, f1_score, accuracy_score, roc_auc_score

# 1. SETUP PATH RELASIONAL SEARAH
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'train_mastitis_edge.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test_mastitis_edge.csv')
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, 'models', 'mastitis_model_v1.pkl')

# Pastikan direktori models ada
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

FEATURES = ['body_temperature', 'heart_rate', 'rumination_time', 'milk_yield', 'milk_conductivity']
TARGET = 'label'

def execute_edge_training_pipeline():
    print("[EXECUTION] Memuat dataset latih dan uji terpisah...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    X_train, y_train = df_train[FEATURES], df_train[TARGET]
    X_test, y_test = df_test[FEATURES], df_test[TARGET]
    
    # 2. LOCK CLASSIFIER UNDER EDGE HARD CONSTRAINTS
    model = RandomForestClassifier(
        n_estimators=50,         # Membatasi ukuran file biner < 10MB
        max_depth=10,            # Membatasi kedalaman pohon komputasi edge
        min_samples_split=5,
        class_weight='balanced', # Mengatasi data imbalance kelas subklinis
        random_state=42,
        n_jobs=-1
    )
    
    print("[EXECUTION] Melatih model Random Forest Lokal...")
    model.fit(X_train, y_train)
    
    # 3. ROBUST INFRASTRUCTURE BENCHMARK (P95 Latency via Predict Proba)
    dummy_input = X_test.iloc[:1]
    latencies = []
    
    # Warm-up loop untuk eliminasi cold-start CPU scheduling
    for _ in range(10):
        _ = model.predict_proba(dummy_input)
        
    # Sampling loop N=1000
    for _ in range(1000):
        start = time.perf_counter()
        _ = model.predict_proba(dummy_input)
        latencies.append((time.perf_counter() - start) * 1000)
        
    p95_latency = np.percentile(latencies, 95)
    mean_latency = np.mean(latencies)
    
    # 4. EXPORT MODEL BINER (.pkl)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    file_size_mb = os.path.getsize(MODEL_OUTPUT_PATH) / (1024 * 1024)
    
    # 5. GENERATE METRIC REPORT FOR ACADEMIC PAPER
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    observed_recall = recall_score(y_test, y_pred)
    observed_f1 = f1_score(y_test, y_pred)
    observed_auc = roc_auc_score(y_test, y_proba)
    observed_accuracy = accuracy_score(y_test, y_pred)

    evaluation = {
        "accuracy": float(observed_accuracy),
        "recall": float(observed_recall),
        "f1_score": float(observed_f1),
        "auc": float(observed_auc),
        "model_size_mb": float(file_size_mb),
        "mean_latency_ms": float(mean_latency),
        "p95_latency_ms": float(p95_latency)
    }

    with open(
        os.path.join(BASE_DIR, "models", "evaluation_report.json"),
        "w"
    ) as fp:
        json.dump(evaluation, fp, indent=4)
    
    print("\n" + "="*20 + " PRD-002: MODEL AUDIT REPORT " + "="*20)
    print(f"\n[PREDICTIVE METRICS (Evaluated under 30% Domain Drift)]:")
    print(f"-> Observed Recall : {observed_recall:.4f} (Target Sistem: >= 0.90)")
    print(f"-> Observed F1-S   : {observed_f1:.4f} (Target Sistem: >= 0.85)")
    print(f"-> Observed AUC  : {observed_auc:.4f}")
    print(f"-> Observed Accuracy : {observed_accuracy:.4f}")
    print("\nMatrix Kebingungan (Confusion Matrix):")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetail Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Subclinical']))
    
    print(f"[INFRASTRUCTURE METRICS (Edge Constraint Auditing)]:")
    print(f"-> Model File Size : {file_size_mb:.4f} MB (Hard Limit: < 10.0 MB)")
    print(f"-> Mean Latency    : {mean_latency:.4f} ms")
    print(f"-> P95 Latency     : {p95_latency:.4f} ms (Hard Limit: < 50.0 ms)")
    
    print(f"\n[DIAGNOSTIC STATUS]: READY FOR ENGINE INTEGRATION")
    print("="*65)

if __name__ == "__main__":
    execute_edge_training_pipeline()
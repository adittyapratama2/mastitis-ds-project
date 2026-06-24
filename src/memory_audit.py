# src/memory_audit.py
# -*- coding: utf-8 -*-
import tracemalloc
import pandas as pd
from engine import LivestockPrescriptiveEngine

def audit_ram_footprint():
    # 1. Mulai pelacakan alokasi memori RAM
    tracemalloc.start()
    
    print("[AUDIT] Memuat model dan menginisialisasi Prescriptive Engine...")
    engine = LivestockPrescriptiveEngine()
    
    # Ambil snapshot pertama setelah engine dimuat ke RAM
    current, peak = tracemalloc.get_traced_memory()
    engine_load_ram = peak / (1024 * 1024) # Konversi ke MB
    
    # 2. Simulasikan beban pemrosesan data
    payload = {
        "body_temperature": 38.9, "heart_rate": 75, 
        "rumination_time": 400, "milk_yield": 24.0, "milk_conductivity": 5.3
    }
    
    print("[AUDIT] Mengeksekusi inferensi + kalkulasi SHAP...")
    for _ in range(100):
        _ = engine.process_reading(payload)
        
    # Ambil snapshot kedua setelah beban kerja selesai
    _, final_peak = tracemalloc.get_traced_memory()
    total_peak_ram = final_peak / (1024 * 1024)
    
    tracemalloc.stop()
    
    print("\n" + "="*20 + " EDGE RAM AUDIT REPORT " + "="*20)
    print(f"-> RAM saat Engine Dimuat : {engine_load_ram:.4f} MB")
    print(f"-> Peak RAM selama Inferensi : {total_peak_ram:.4f} MB (Hard Limit: < 100.0 MB)")
    print(f"-> Status Kelayakan Edge   : {'AMAN' if total_peak_ram < 100.0 else 'OVER_BUDGET'}")
    print("="*63)

if __name__ == "__main__":
    audit_ram_footprint()
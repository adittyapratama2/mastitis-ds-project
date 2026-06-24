# src/engine.py
# -*- coding: utf-8 -*-
import joblib
import pandas as pd
import numpy as np
import shap
import time
import os

class LivestockPrescriptiveEngine:
    CRITICAL_TEMP_THRES = 40.0
    CRITICAL_COND_THRES = 6.5
    
    LOW_RISK_THRES = 0.30
    HIGH_RISK_THRES = 0.70
    
    PRESCRIPTION_LOOKUP = {
        "MONITOR_RUTIN": {"message": "Sapi dalam kondisi sehat. Lanjutkan protokol pemeliharaan standar.", "timeline": "24 jam", "urgency": "LOW"},
        "CMT_TEST": {"message": "Indikasi awal subclinical mastitis terdeteksi. Diperlukan tindakan pengujian tambahan.", "timeline": "6 jam", "urgency": "MEDIUM"},
        "VET_CONSULT": {"message": "Risiko tinggi subclinical mastitis. Diperlukan isolasi dan intervensi medis.", "timeline": "12 jam", "urgency": "HIGH"},
        "CLINICAL_BYPASS": {"message": "KRITIS! Batas klinis terlampaui. Isolasi darurat dan panggil dokter hewan darurat sekarang!", "timeline": "1 jam", "urgency": "CRITICAL"}
    }

    # Poin Koreksi 1: Pemisahan Rekomendasi Berlapis Berdasarkan Risk Tier (Medical Consistency)
    SPECIFIC_MITIGATION_LOOKUP = {
        "LOW": {
            "body_temperature": "Pantau kestabilan suhu lingkungan kandang dan pastikan kecukupan hidrasi air minum sapi.",
            "milk_conductivity": "Lanjutkan sanitasi teat dipping pasca-pemerahan dan monitoring rutin kualitas fisik susu harian.",
            "rumination_time": "Pertahankan kualitas dan jadwal pemberian pakan serat (hijauan) berkualitas tinggi.",
            "milk_yield": "Pertahankan manajemen nutrisi pakan untuk menjaga stabilitas fluks produksi susu harian.",
            "heart_rate": "Lanjutkan monitoring kenyamanan lingkungan untuk menjaga kondisi sapi tetap rileks."
        },
        "MEDIUM": {
            "body_temperature": "Periksa adanya tanda-tanda stres panas (heat stress) awal dan lakukan observasi suhu berkala.",
            "milk_conductivity": "Lakukan California Mastitis Test (CMT) mandiri pada kuartir ambing dan inspeksi kebersihan puting.",
            "rumination_time": "Evaluasi palatabilitas pakan serat dan amati potensi penurunan aktivitas kunyah sapi.",
            "milk_yield": "Catat riwayat fluktuasi volume perahan harian untuk mengantisipasi penurunan drastis.",
            "heart_rate": "Amati tanda-tanda kegelisahan atau penurunan nafsu makan pada sapi."
        },
        "HIGH": {
            "body_temperature": "Periksa kemungkinan infeksi sistemik akut, bersihkan ventilasi, dan lakukan kompres air hangat.",
            "milk_conductivity": "Lakukan CMT test segera, pisahkan fluks susu terinfeksi, dan jadwalkan konsultasi veteriner.",
            "rumination_time": "Evaluasi potensi gangguan pencernaan akut (asidosis) dan kurangi faktor stres fisik lingkungan.",
            "milk_yield": "Segera periksa kondisi fisik ambing terhadap indikasi pembengkakan atau pengerasan jaringan.",
            "heart_rate": "Segera isolasi sapi untuk meminimalkan cedera fisik akibat rasa nyeri internal."
        }
    }

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(os.path.dirname(current_dir), 'models', 'mastitis_model_v1.pkl')
        
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.features = ['body_temperature', 'heart_rate', 'rumination_time', 'milk_yield', 'milk_conductivity']

    def process_reading(self, raw_data: dict) -> dict:
        start_time = time.perf_counter()
        
        # VALIDASI INPUT
        missing_fields = [field for field in self.features if field not in raw_data]
        if missing_fields:
            raise ValueError(f"Data kontrak tidak lengkap. Kolom hilang: {missing_fields}")

        # SAFETY ALERT LAYER
        if (raw_data['body_temperature'] >= self.CRITICAL_TEMP_THRES) or (raw_data['milk_conductivity'] >= self.CRITICAL_COND_THRES):
            elapsed_time = (time.perf_counter() - start_time) * 1000
            return {
                "prediction": "Clinical Mastitis Anomaly",
                "probability": 1.0,
                "confidence": 0.9990, # Poin Koreksi 3: Clamp 99.9%
                "risk_tier": "CRITICAL",
                "root_cause": "Critical Operational Limit Breach",
                "root_cause_impact": 1.0,
                "action_code": "CLINICAL_BYPASS",
                "inference_time_ms": round(elapsed_time, 2),
                "meta": self.PRESCRIPTION_LOOKUP["CLINICAL_BYPASS"]
            }

        # MACHINE LEARNING PATH
        input_df = pd.DataFrame([raw_data])[self.features]
        prob = self.model.predict_proba(input_df)[0][1]
        
        # Poin Koreksi 3: Clamp Model Confidence Maksimal 99.9% (Realistic Uncertainty)
        raw_confidence = max(prob, 1.0 - prob)
        confidence = min(raw_confidence, 0.9990)
        
        # SHAP COMPATIBILITY HANDLING
        try:
            shap_values = self.explainer(input_df)
            if hasattr(shap_values, "values"):
                shap_contribs = shap_values.values[0, :, 1] if len(shap_values.values.shape) == 3 else shap_values.values[0]
            else:
                shap_contribs = shap_values[1][0]
        except Exception:
            shap_contribs = np.zeros(len(self.features))
            
        top_feature_idx = np.argmax(np.abs(shap_contribs))
        top_feature = self.features[top_feature_idx]
        top_contrib_impact = float(np.abs(shap_contribs[top_feature_idx]))
        
        # RISK STRATIFICATION MATRIX
        if prob < self.LOW_RISK_THRES:
            action_code, tier = "MONITOR_RUTIN", "LOW"
        elif prob < self.HIGH_RISK_THRES:
            action_code, tier = "CMT_TEST", "MEDIUM"
        else:
            action_code, tier = "VET_CONSULT", "HIGH"
            
        base_prescription = self.PRESCRIPTION_LOOKUP[action_code]
        
        # Ambil rekomendasi berlapis dinamis (Poin Koreksi 1)
        specific_action = self.SPECIFIC_MITIGATION_LOOKUP[tier].get(top_feature, "")
        
        # Poin Koreksi 4: Semantik Root Cause Saat Sapi Sehat (LOW Risk)
        if tier == "LOW":
            cleaned_feature_name = "Stable Sensor Profile"
            top_contrib_impact = 0.0000 # Tidak ada dampak pembengkakan fitur jika profil stabil
        else:
            cleaned_feature_name = top_feature.replace('_', ' ').title()
            
        elapsed_time = (time.perf_counter() - start_time) * 1000
        
        # Poin Koreksi 2: Buka Presisi Desimal (Probabilitas dikembalikan dalam float murni, UI yang formatting %)
        return {
            "prediction": "Suspected Subclinical Mastitis" if prob >= self.LOW_RISK_THRES else "Healthy Stable Profile",
            "probability": round(float(prob), 4),
            "confidence": round(float(confidence), 4),
            "risk_tier": tier,
            "root_cause": cleaned_feature_name,
            "root_cause_impact": round(top_contrib_impact, 4),
            "action_code": action_code,
            "inference_time_ms": round(elapsed_time, 2),
            "meta": {
                "message": f"{base_prescription['message']}\n\nTindakan Spesifik (XAI Grounded): {specific_action}",
                "timeline": base_prescription["timeline"],
                "urgency": base_prescription["urgency"]
            }
        }
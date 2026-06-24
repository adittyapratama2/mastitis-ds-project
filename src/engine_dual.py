# src/engine_dual.py
# -*- coding: utf-8 -*-
"""
Master's Thesis Artifact - S2 Big Data Amikom Yogyakarta
Author: Adittya Pratama & Team
Description: HYBRID PRODUCTION INFERENCE ENGINE - Multi-Scenario Framework 
             with Clinical Safety Bypass, SHAP XAI, and Dynamic Prescriptive Triage.
"""

import os
import time
import joblib
import pandas as pd
import numpy as np
import shap

class MastitisEdgeEngineDual:
    # 1. HARDWARE-LEVEL CRITICAL MEDICAL THRESHOLDS (Safety Net Layer)
    CRITICAL_TEMP_THRES = 40.0
    CRITICAL_COND_THRES = 6.5
    
    # 2. RISK STRATIFICATION THRESHOLDS
    LOW_RISK_THRES = 0.30
    HIGH_RISK_THRES = 0.70

    def __init__(self):
        self.model_dir = 'models'
        self.features = ['body_temperature', 'heart_rate', 'rumination_time', 'milk_yield', 'milk_conductivity']
        self.models = {}
        self.explainers = {}
        self._preload_dual_scenarios()

    def _preload_dual_scenarios(self):
        """Memuat biner model dan menginisialisasi SHAP Explainer secara paralel di memori RAM"""
        for sc in ['empirical', 'synthetic']:
            model_path = os.path.join(self.model_dir, f'mastitis_model_{sc}.pkl')
            if os.path.exists(model_path):
                # Load Model Biner
                model = joblib.load(model_path)
                self.models[sc] = model
                
                # Inisialisasi TreeExplainer Ter-cache (Anti I/O Degradation)
                self.explainers[sc] = shap.TreeExplainer(model)
            else:
                self.models[sc] = None
                self.explainers[sc] = None

    def process_inference(self, scenario, body_temp, heart_rate, rumination, milk_yield, milk_conductivity):
        """
        Mengeksekusi Pipeline Inferensi Komprehensif: 
        Safety Bypass -> ML Classification -> SHAP XAI Atribusi -> Prescriptive Matrix
        """
        start_time = time.time()
        
        # Kontrak payload DataFrame searah
        payload = {
            'body_temperature': float(body_temp),
            'heart_rate': int(heart_rate),
            'rumination_time': int(rumination),
            'milk_yield': float(milk_yield),
            'milk_conductivity': float(milk_conductivity)
        }
        eval_df = pd.DataFrame([payload])[self.features]

        # =============================================================
        # JALUR 1: CLINICAL SAFETY BYPASS (Penyelamat Nyawa Sapi)
        # =============================================================
        if payload['body_temperature'] >= self.CRITICAL_TEMP_THRES or payload['milk_conductivity'] >= self.CRITICAL_COND_THRES:
            inference_time = round((time.time() - start_time) * 1000, 2)
            return self._trigger_clinical_bypass(payload, inference_time)

        # Validasi Keberadaan Model
        model = self.models.get(scenario)
        explainer = self.explainers.get(scenario)
        if model is None:
            return {"status": "ERROR", "message": f"Berkas model untuk skenario '{scenario}' tidak aktif!"}

        # =============================================================
        # JALUR 2: MACHINE LEARNING INFERENCE & CONFIDENCE CLAMPING
        # =============================================================
        prob_raw = model.predict_proba(eval_df)[0][1]
        prediction = 1 if prob_raw >= 0.5 else 0
        
        # Confidence Clamping Pattern (Proteksi Presentasi dari Angka Sempurna Palsu)
        confidence = min(prob_raw, 0.9990) if prediction == 1 else min(1 - prob_raw, 0.9990)

        # =============================================================
        # JALUR 3: EXPLAINABLE AI (LOCAL SHAP ATTRIBUTION TRACKING)
        # =============================================================
        try:
            shap_values = explainer(eval_df)
            if len(shap_values.values.shape) == 3:
                shap_contribs = shap_values.values[0, :, 1]
            else:
                shap_contribs = shap_values.values[0]
            
            # Cari fitur pemicu utama berdasarkan nilai koefisien absolut SHAP tertinggi
            root_cause_idx = np.argmax(np.abs(shap_contribs))
            root_cause_var = self.features[root_cause_idx]
            root_cause_impact = round(shap_contribs[root_cause_idx], 4)
        except Exception:
            # Safe Guardrail Fallback jika kalkulasi biner SHAP crash
            root_cause_var = "milk_conductivity"
            root_cause_impact = 0.0

        # =============================================================
        # JALUR 4: RISK STRATIFICATION & DYNAMIC PRESCRIPTIVE ACTION
        # =============================================================
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        # Penentuan Risk Tier berdasarkan batas ambang probabilitas
        if prob_raw >= self.HIGH_RISK_THRES:
            risk_tier = "HIGH"
            action_code = "ISOLATE_AND_TEST"
            urgency = "TINGGI (Segera Tindak)"
            timeline = "Dalam waktu < 3 Jam"
            message = self._generate_specific_prescription(root_cause_var, risk_tier)
        elif prob_raw >= self.LOW_RISK_THRES:
            risk_tier = "MEDIUM"
            action_code = "MONITOR_AND_SANANITIZE"
            urgency = "SEDANG (Waspada)"
            timeline = "Sesi perah berikutnya"
            message = self._generate_specific_prescription(root_cause_var, risk_tier)
        else:
            risk_tier = "LOW"
            action_code = "ROUTINE_MAINTENANCE"
            urgency = "RENDAH (Aman)"
            timeline = "Berkala (Mingguan)"
            message = "Kondisi fisiologis dan parameter produksi susu sapi stabil dalam batas normal harian. Lanjutkan pemeliharaan rutin."

        return {
            "status": "SUCCESS",
            "prediction": prediction,
            "probability": prob_raw,
            "confidence": confidence,
            "risk_tier": risk_tier,
            "action_code": action_code,
            "root_cause": root_cause_var.replace('_', ' ').title(),
            "root_cause_impact": root_cause_impact,
            "inference_time_ms": inference_time,
            "meta": {"urgency": urgency, "timeline": timeline, "message": message},
            "features_used": payload
        }

    def _trigger_clinical_bypass(self, payload, inference_time):
        """Metode bypass otomatis ketika mendeteksi anomali ekstrem penanda bahaya"""
        return {
            "status": "SUCCESS",
            "prediction": 1,
            "probability": 0.9990,
            "confidence": 0.9990,
            "risk_tier": "CRITICAL",
            "action_code": "CLINICAL_BYPASS",
            "root_cause": "Suhu/Konduktivitas Ekstrem",
            "root_cause_impact": 1.0,
            "inference_time_ms": inference_time,
            "meta": {
                "urgency": "GAWAT DARURAT (VETERINER Kritis)",
                "timeline": "Seketika / Instan",
                "message": "🚨 BYPASS MEDIS DIPICU! Terdeteksi parameter kritis di luar batas toleransi biologis hewan (Suhu ≥ 40°C atau Konduktivitas ≥ 6.5 mS/cm). SEGERA ISOLASI SAPI, HENTIKAN KONSUMSI SUSU PERAHAN EKOR INI, dan panggil Dokter Hewan Spesialis!"
            },
            "features_used": payload
        }

    def _generate_specific_prescription(self, root_cause, risk_tier):
        """Rule-Engine untuk melahirkan instruksi spesifik dinamis berdasarkan root cause SHAP"""
        prescriptions = {
            "body_temperature": {
                "HIGH": "🚨 Indikasi demam sistemik tinggi terdeteksi. Berikan kompres air hangat pada area inguinal, pindahkan sapi ke area isolasi ber-ventilasi kuat, berikan hidrasi ad-libitum, dan siapkan injeksi antipiretik.",
                "MEDIUM": "⚠️ Kenaikan suhu tubuh ringan. Lakukan pengecekan manual termometer rektal untuk konfirmasi, bersihkan lingkungan sekeliling tempat istirahat (stall), dan pantau indeks cekaman panas (THI)."
            },
            "milk_conductivity": {
                "HIGH": "🚨 Lonjakan konduktivitas listrik ekstrim menandakan lisis sel ambing. Lakukan tes CMT (California Mastitis Test) seketika pada keempat kuartir ambing, lakukan dipping pasca-perah menggunakan iodin 1%, dan pisahkan wadah tampung susu.",
                "MEDIUM": "⚠️ Konduktivitas susu berada di ambang batas atas. Tingkatkan higienitas prosedur pre-milking (cuci tangan operator & desinfeksi teat cup), dan lakukan stripping manual sebelum mesin dipasang."
            },
            "milk_yield": {
                "HIGH": "🚨 Penurunan produksi susu drop drastis secara mendadak. Indikasi kuat kerusakan kelenjar sekretori. Berikan pakan konsentrat tambahan energi, periksa adanya pembengkakan fisik atau rasa nyeri pada ambing.",
                "MEDIUM": "⚠️ Penurunan produksi minor harian. Evaluasi palatabilitas pakan total campuran (TMR) dan pastikan ketersediaan air minum tidak terhambat."
            },
            "heart_rate": {
                "HIGH": "🚨 Detak jantung tinggi mengindikasikan stres akut atau rasa nyeri hebat pada hewan. Lakukan pemeriksaan fisik menyeluruh untuk mendeteksi tanda-tanda trauma fisik, pincang (lameness), atau distres pernapasan.",
                "MEDIUM": "⚠️ Detak jantung meningkat di atas baseline normal harian. Evaluasi kondisi sirkulasi udara lingkungan terhadap potensi heat stress atau gangguan kenyamanan minor pada area kandang."
            },
            "rumination_time": {
                "HIGH": "🚨 Penurunan drastis waktu ruminasi mengindikasikan gangguan pencernaan akut atau inflamasi sistemik hebat. Segera pisahkan sapi, lakukan evaluasi pakan total, dan periksa kemungkinan adanya asidosis rumen akut (ARA).",
                "MEDIUM": "⚠️ Waktu ruminasi/mengunyah menurun. Cek kualitas serat kasar pada pakan hijauan, pastikan stabilitas asupan bahan kering (DMI), dan pantau tanda awal kelesuan."
            }
        }
        
        fallback_high = "🚨 Risiko tinggi infeksi subklinis. Ambil sampel susu untuk uji laboratorium, perketat biosekuriti gerbang perahan, dan lakukan dipping ambing pasca-perah."
        fallback_med = "⚠️ Peringatan awal anomali perilaku. Tingkatkan pengawasan intensif terhadap asupan pakan dan kebersihan alas tidur sapi selama 24 jam ke depan."
        
        return prescriptions.get(root_cause, {}).get(risk_tier, fallback_high if risk_tier == "HIGH" else fallback_med)
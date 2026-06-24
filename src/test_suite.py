# src/test_suite.py
# -*- coding: utf-8 -*-
"""
Master's Thesis Artifact - S2 Big Data Amikom Yogyakarta
Author: Adittya Pratama & Team
Description: Automated Rigorous Test Suite for Dual-Scenario Inference Engine.
             Includes Safety Bypass, Risk Stratification, and Latency Stress Test.
"""

import unittest
import numpy as np
import time
from engine_dual import MastitisEdgeEngineDual

class CoreEngineRigorousTestSuite(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Inisialisasi engine dual-scenario satu kali untuk seluruh pengujian.
        Default diuji pada skenario 'synthetic' (skenario terkuat dengan Domain Drift).
        """
        cls.engine = MastitisEdgeEngineDual()
        cls.scenario = 'synthetic' # Menguji ketahanan skenario utama tesis

    def test_tc001_boundary_safety_alert_layer(self):
        """
        [FUNCTIONAL] Menguji akurasi trigger Clinical Bypass tepat pada batas threshold.
        - Di bawah batas bypass (6.4) -> harus lewat jalur ML (action_code bukan CLINICAL_BYPASS)
        - Tepat pada batas konduktivitas (6.5) -> harus memotong jalur ML dan memicu bypass klinis.
        """
        # Kasus 1: Tepat di bawah batas bypass (6.4) -> Harus diproses oleh Machine Learning
        res_under = self.engine.process_inference(
            scenario=self.scenario,
            body_temp=39.9,
            heart_rate=85,
            rumination=300,
            milk_yield=20.0,
            milk_conductivity=6.4
        )
        self.assertNotEqual(res_under["action_code"], "CLINICAL_BYPASS")
        self.assertEqual(res_under["status"], "SUCCESS")
        
        # Kasus 2: Tepat pada batas konduktivitas bypass (6.5) -> Harus memicu Critical Bypass Instan
        res_exact = self.engine.process_inference(
            scenario=self.scenario,
            body_temp=38.5,
            heart_rate=70,
            rumination=480,
            milk_yield=25.0,
            milk_conductivity=6.5
        )
        self.assertEqual(res_exact["action_code"], "CLINICAL_BYPASS")
        self.assertEqual(res_exact["risk_tier"], "CRITICAL")
        self.assertEqual(res_exact["prediction"], 1)

    def test_tc002_risk_stratification_tiers(self):
        """
        [FUNCTIONAL] Memastikan pembagian kelas risiko (LOW) konsisten pada sampel sehat.
        Profil fisiologis normal harus masuk ke dalam klasifikasi tingkat risiko LOW.
        """
        res_healthy = self.engine.process_inference(
            scenario=self.scenario,
            body_temp=38.4,
            heart_rate=68,
            rumination=500,
            milk_yield=26.5,
            milk_conductivity=4.6
        )
        self.assertEqual(res_healthy["risk_tier"], "LOW")
        self.assertEqual(res_healthy["prediction"], 0) # Label Sehat

    def test_tc003_stress_and_latency_stability(self):
        """
        [PERFORMANCE] Menguji stabilitas latensi komputasi lokal di bawah beban beruntun.
        Mensimulasikan fluks data masuk untuk memastikan P95 Latency tetap di bawah batas edge (< 50 ms).
        """
        base_payload = {
            "body_temp": 38.9,
            "heart_rate": 78,
            "rumination": 380,
            "milk_yield": 22.0,
            "milk_conductivity": 5.2
        }
        
        latencies = []
        n_iter = 500
        
        # Simulasikan semburan 500 data telemetri IoT masuk berturut-turut
        for _ in range(n_iter):
            start = time.perf_counter()
            _ = self.engine.process_inference(
                scenario=self.scenario,
                **base_payload
            )
            latencies.append((time.perf_counter() - start) * 1000) # Konversi ke milidetik
            
        p95_latency = np.percentile(latencies, 95)
        mean_latency = np.mean(latencies)
        
        print(f"\n[STRESS TEST RESULT] N={n_iter} Skenario: {self.scenario.upper()}")
        print(f"      |-- Mean Inference Latency : {mean_latency:.4f} ms")
        print(f"      |-- P95 Inference Latency  : {p95_latency:.4f} ms (Target: < 50.0 ms)")
        
        # Pengujian asersi performa komputasi lokal perangkat keras edge
        self.assertTrue(p95_latency < 50.0, 
                        f"P95 Latency membengkak hingga {p95_latency:.2f} ms (melewati batas kritis 50 ms!)")
        self.assertTrue(mean_latency < 20.0, 
                        f"Mean Latency melambat hingga {mean_latency:.2f} ms (harus di bawah 20 ms)")

    def test_tc004_cross_scenario_availability(self):
        """
        [CROSS-VALIDATION] Memverifikasi ketersediaan kedua berkas biner model 
        (Empirical & Synthetic) di dalam memori inti engine.
        """
        self.assertIsNotNone(self.engine.models.get('empirical'), "Model biner 'empirical' tidak termuat di memori!")
        self.assertIsNotNone(self.engine.models.get('synthetic'), "Model biner 'synthetic' tidak termuat di memori!")

if __name__ == "__main__":
    print("=============================================================")
    print("=== RUNNING AUTOMATED RIGOROUS CORE ENGINE TEST SUITE ===")
    print("=============================================================")
    unittest.main()
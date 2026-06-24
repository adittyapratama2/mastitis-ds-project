# src/app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from engine import LivestockPrescriptiveEngine

# 1. LAYER MEMORI: Caching Engine Instance (Anti I/O Degradation)
@st.cache_resource
def load_cached_engine():
    return LivestockPrescriptiveEngine()

try:
    engine = load_cached_engine()
except Exception as e:
    st.error(f"Gagal memuat biner model. Error: {e}")
    st.stop()

# Konfigurasi Tema Utama Antarmuka Streamlit
st.set_page_config(page_title="Edge Prescriptive DSS", layout="wide")
st.title("🐄 Resource-Constrained Early Mastitis Decision Support System")
st.write("---")

# Membagi Layout Menjadi 2 Kolom Tegas (Form Input vs Center Dashboard)
col_form, col_dashboard = st.columns([2, 3])

# 2. PRESENTATION LAYER: FORM INPUT (State Isolation Pattern via st.form)
with col_form:
    st.header("IoT Edge Telemetry Input")
    with st.form("sensor_ingestion_form"):
        body_temp = st.slider("Body Temperature (°C)", 37.0, 41.5, 38.5, 0.1)
        heart_rate = st.number_input("Heart Rate (bpm)", 50, 120, 70, 1)
        rumination_time = st.slider("Rumination Time (min/day)", 100, 600, 480, 5)
        milk_yield = st.number_input("Milk Yield (liters/day)", 2.0, 40.0, 25.0, 0.5)
        milk_cond = st.slider("Electrical Conductivity (mS/cm)", 4.0, 10.0, 4.8, 0.1)
        
        submit_btn = st.form_submit_button("Eksekusi Analisis Klinis")

# 3. APPLICATION LAYER: PROCESS DATA RESPONSE (BLOK INTEGRASI UTAMA)
if submit_btn:
    payload = {
        "body_temperature": body_temp,
        "heart_rate": heart_rate,
        "rumination_time": rumination_time,
        "milk_yield": milk_yield,
        "milk_conductivity": milk_cond
    }
    
    # Eksekusi pipeline logika di core engine terisolasi
    res = engine.process_reading(payload)
    
    with col_dashboard:
        st.header("Prescriptive Diagnostic Center")
        
        # Rendering Alert UI Berdasarkan Stratifikasi Tingkat Risiko (Triage Alert Layer)
        if res["risk_tier"] == "CRITICAL":
            st.error(f"🔴 DIAGNOSIS: {res['prediction']}")
        elif res["risk_tier"] == "HIGH":
            st.warning(f"🟠 DIAGNOSIS: {res['prediction']}")
        elif res["risk_tier"] == "MEDIUM":
            st.info(f"🟡 DIAGNOSIS: {res['prediction']}")
        else:
            st.success(f"🟢 DIAGNOSIS: {res['prediction']}")
            
        # Display Triple-Grid Metrik Sinkron (Sesuai Hasil Review Produksi)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            # Poin Koreksi Proteksi Presentasi: Clamp nilai probabilitas visual maksimal 99.90%
            display_prob = min(res["probability"], 0.9990)
            st.metric(label="Probability Score", value=f"{display_prob * 100:.2f}%")
        with col_m2:
            # Menampilkan Konfidensi Ter-clamp Realistis maksimal 99.90%
            st.metric(label="Model Confidence", value=f"{res['confidence'] * 100:.2f}%")
        with col_m3:
            st.metric(label="Edge Inference Latency", value=f"{res['inference_time_ms']} ms")
            
        # Deskripsi Kualitatif Root Cause & Dampak Pengaruh SHAP Aktual
        st.markdown(f"**Root Cause Variable:** `{res['root_cause']}` | **SHAP Impact Weight:** `{res['root_cause_impact']}`")
        st.markdown(f"**Tingkat Urgensi:** `{res['meta']['urgency']}` | **Batas Waktu Mitigasi:** `{res['meta']['timeline']}`")
        
        # Tampilkan Kotak Instruksi Tindakan Preskriptif Lapangan
        # GANTI DENGAN BLOK KONDISIONAL DINAMIS INI:
        st.markdown("**Instruksi Preskriptif Tindakan Lapangan:**")
        if res["risk_tier"] == "CRITICAL":
            st.error(res['meta']['message'])    # Warna Merah untuk Kondisi Kritis Akut
        elif res["risk_tier"] == "HIGH":
            st.warning(res['meta']['message'])  # Warna Jingga/Tua untuk Risiko Tinggi
        elif res["risk_tier"] == "MEDIUM":
            st.info(res['meta']['message'])     # Warna Kuning/Biru Muda (Tergantung Tema) untuk Peringatan Awal
        else:
            st.success(res['meta']['message'])  # Warna Hijau HANYA jika sapi benar-benar sehat (LOW)
        
        # 4. EXPLAINABLE AI VISUALIZATION (Hanya muncul jika lolos dari safety bypass)
        if res["action_code"] != "CLINICAL_BYPASS":
            st.write("---")
            st.subheader("Explainable AI (XAI): Atribusi Kontribusi Variabel")
            
            # Mempersiapkan DataFrame evaluasi searah
            eval_df = pd.DataFrame([payload])[engine.features]
            
            try:
                # Menggunakan instance explainer yang sudah ter-cache aman di memori engine
                shap_values = engine.explainer(eval_df)
                
                # Handling robust multi-class array untuk ekstraksi nilai kelas target (Subclinical = Indeks 1)
                if hasattr(shap_values, "values"):
                    if len(shap_values.values.shape) == 3:
                        shap_contribs = shap_values.values[0, :, 1]
                    else:
                        shap_contribs = shap_values.values[0]
                else:
                    # Fallback objek list array legacy
                    shap_contribs = shap_values[1][0]
            except Exception:
                # Safe guardrail agar UI tidak crash jika backend mengalami kegagalan kalkulasi
                shap_contribs = [0.0] * len(engine.features)
                
            # Konfigurasi Kanvas Grafik Standar Jurnal Internasional (Camera-Ready Plot)
            plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
            plt.rcParams['font.size'] = 9
            plt.rcParams['axes.unicode_minus'] = False # Mencegah error render tanda minus
            
            fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
            
            # Plot Bar Horizontal Kontribusi Lokal SHAP
            shap.bar_plot(
                shap_contribs, 
                feature_names=[f.replace('_', ' ').title() for f in engine.features],
                show=False,
                max_display=5
            )
            
            plt.title("Local Feature Contribution (Subclinical Target Class)", fontsize=10, fontweight='bold', pad=12)
            plt.xlabel("SHAP Value (Impact Contribution Coefficient)")
            plt.tight_layout()
            
            # Menampilkan Grafik Matplotlib Terisolasi di UI Streamlit
            st.pyplot(fig)
            
            # Memory Management Guard: Hapus canvas dan close figure dari memori RAM setelah dirender
            plt.clf()
            plt.close(fig)
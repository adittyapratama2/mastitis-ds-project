# src/app_dual.py
# -*- coding: utf-8 -*-
"""
Master's Thesis Artifact - S2 Big Data Amikom Yogyakarta
Author: Adittya Pratama & Team
Description: Dual-Scenario Dashboard with SHAP Explainability & Safety Bypass.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from engine_dual import MastitisEdgeEngineDual
import os

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Mastitis Edge DSS - Dual Scenario",
    layout="wide",
    page_icon="🐄"
)

# ==========================================
# CACHE ENGINE (HANYA DILOAD SEKALI)
# ==========================================
@st.cache_resource
def get_engine():
    return MastitisEdgeEngineDual()

engine = get_engine()

# ==========================================
# SIDEBAR: PEMILIHAN SKENARIO & AUDIT MODEL
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Perangkat Edge")

selected_option = st.sidebar.selectbox(
    "Pilih Skenario Evaluasi Data:",
    options=[
        "Skenario A: Data Lapangan Empiris (Tanpa Wearable Riil)",
        "Skenario B: Data Multi-Sensor Terkalibrasi (Sintetis + Drift)"
    ],
    index=0
)

# Parse scenario key
scenario_key = "empirical" if "Empiris" in selected_option else "synthetic"

st.sidebar.info(f"Model Aktif: **mastitis_model_{scenario_key}.pkl**")

# Tampilkan ukuran file model sebagai audit edge computing
model_path = os.path.join('models', f'mastitis_model_{scenario_key}.pkl')
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    st.sidebar.success(f"💾 File Size Model: {size_mb:.4f} MB (< 10 MB)")
else:
    st.sidebar.error("❌ Berkas model .pkl tidak ditemukan! Jalankan `python src/train_dual.py`")

st.sidebar.markdown("---")
st.sidebar.caption("📌 Pastikan model telah dilatih dengan `train_dual.py` sebelum menggunakan dashboard.")

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("🐄 Local Prescriptive Livestock Analytics Platform")
st.subheader(f"Skenario Aktif: **{selected_option}**")
st.write("---")

# Layout dua kolom: input sensor (kiri) dan hasil analisis (kanan)
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("### 📡 Simulator Input Telemetri IoT")
    st.write("Masukkan nilai sensor untuk simulasi kondisi sapi saat ini:")

    # Slider dan input dengan batas biologis yang sesuai
    body_temp = st.slider("Suhu Tubuh / Susu (°C)", min_value=34.0, max_value=41.5, value=38.5, step=0.1)
    heart_rate = st.slider("Detak Jantung (bpm)", min_value=45, max_value=130, value=70, step=1)
    rumination = st.slider("Waktu Ruminasi (menit/hari)", min_value=100, max_value=650, value=480, step=5)
    milk_yield = st.number_input("Produksi Susu Harian (Kg)", min_value=0.0, max_value=50.0, value=20.0, step=0.5)
    milk_conductivity = st.number_input("Konduktivitas Listrik Susu (mS/cm)", min_value=2.0, max_value=12.0, value=4.5, step=0.1)

    btn_execute = st.button("🚀 Jalankan Inferensi Lokal", use_container_width=True)

# ==========================================
# KOLOM KANAN: HASIL ANALISIS & VISUALISASI XAI
# ==========================================
with col_right:
    st.markdown("### 📊 Hasil Analisis Preskriptif Lokal")

    if btn_execute:
        # Panggil engine dual dengan metode process_inference
        res = engine.process_inference(
            scenario=scenario_key,
            body_temp=body_temp,
            heart_rate=heart_rate,
            rumination=rumination,
            milk_yield=milk_yield,
            milk_conductivity=milk_conductivity
        )

        if res["status"] == "SUCCESS":
            # --- Tampilkan metrik utama ---
            prob_pct = res["probability"] * 100
            confidence_pct = res["confidence"] * 100

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(
                    label="Probabilitas Mastitis",
                    value=f"{prob_pct:.2f}%",
                    delta="⚠️ Risiko" if res["prediction"] == 1 else "✅ Aman"
                )
            with col_m2:
                st.metric(label="Keyakinan Model", value=f"{confidence_pct:.2f}%")
            with col_m3:
                st.metric(label="Waktu Inferensi", value=f"{res['inference_time_ms']} ms")

            # --- Triage dan warna sesuai tingkat risiko ---
            risk_tier = res["risk_tier"]
            if risk_tier == "CRITICAL":
                st.error(f"🔴 **Triage: {risk_tier} (GAWAT DARURAT)**")
            elif risk_tier == "HIGH":
                st.warning(f"🟠 **Triage: {risk_tier} (Segera Tindak)**")
            elif risk_tier == "MEDIUM":
                st.info(f"🟡 **Triage: {risk_tier} (Waspada)**")
            else:
                st.success(f"🟢 **Triage: {risk_tier} (Aman)**")

            # --- Root cause & dampak SHAP ---
            st.markdown(f"**Root Cause (Fitur Paling Berpengaruh):** `{res['root_cause']}`")
            st.markdown(f"**Dampak SHAP:** `{res['root_cause_impact']}`")

            # --- Instruksi preskriptif (dinamis) ---
            st.markdown("#### 📋 Instruksi Klinis Lapangan:")
            msg = res["meta"]["message"]
            if risk_tier == "CRITICAL":
                st.error(msg)
            elif risk_tier == "HIGH":
                st.warning(msg)
            elif risk_tier == "MEDIUM":
                st.info(msg)
            else:
                st.success(msg)

            # --- Detail tambahan dari meta ---
            st.caption(f"⏳ Timeline: {res['meta']['timeline']}  |  🏷️ Urgensi: {res['meta']['urgency']}")

            # --- VISUALISASI XAI (SHAP) ---
            # Hanya tampilkan jika bukan clinical bypass (karena bypass tidak melalui ML)
            if res["action_code"] != "CLINICAL_BYPASS":
                st.write("---")
                st.subheader("🧠 Explainable AI (SHAP): Kontribusi Lokal Fitur")

                # Ambil explainer yang sudah ter-cache di engine
                explainer = engine.explainers.get(scenario_key)
                if explainer is not None:
                    # Buat dataframe dari payload (sama seperti di engine)
                    payload = {
                        'body_temperature': body_temp,
                        'heart_rate': heart_rate,
                        'rumination_time': rumination,
                        'milk_yield': milk_yield,
                        'milk_conductivity': milk_conductivity
                    }
                    eval_df = pd.DataFrame([payload])[engine.features]

                    try:
                        shap_values = explainer(eval_df)
                        # Ekstrak kontribusi untuk kelas 1 (subclinical)
                        if len(shap_values.values.shape) == 3:
                            shap_contribs = shap_values.values[0, :, 1]
                        else:
                            shap_contribs = shap_values.values[0]

                        # Plot dengan matplotlib
                        plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
                        plt.rcParams['font.size'] = 9
                        plt.rcParams['axes.unicode_minus'] = False

                        fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
                        shap.bar_plot(
                            shap_contribs,
                            feature_names=[f.replace('_', ' ').title() for f in engine.features],
                            show=False,
                            max_display=5
                        )
                        plt.title("Kontribusi Fitur Lokal (Kelas Subklinis)", fontsize=10, fontweight='bold')
                        plt.xlabel("Nilai SHAP (Dampak)")
                        plt.tight_layout()

                        st.pyplot(fig)
                        plt.clf()
                        plt.close(fig)

                    except Exception as e:
                        st.warning(f"Visualisasi SHAP gagal: {e}. Tampilan hanya hasil prediksi.")

                else:
                    st.info("Explainer tidak tersedia untuk skenario ini.")

            else:
                st.info("⚠️ Clinical Bypass aktif – visualisasi SHAP tidak dihasilkan karena model tidak digunakan.")

        else:
            st.error(f"❌ Error: {res.get('message', 'Unknown error')}")

    else:
        st.info("🔍 Masukkan nilai sensor dan tekan tombol di sebelah kiri untuk memulai analisis.")
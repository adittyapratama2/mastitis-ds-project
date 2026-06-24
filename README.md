# Resource-Constrained Early Mastitis Decision Support System (DSS) 🐄

Sistem Penunjang Keputusan preskriptif berbasis *Edge Computing* untuk deteksi dini Mastitis Subklinis pada sapi perah. Menggunakan **Random Forest** dengan **Explainable AI (SHAP)** dan **Dual-Scenario Validation** yang dioptimalkan untuk perangkat komputasi lokal berspesifikasi rendah.

---

## 📌 Fitur Utama & Keunggulan Sistem

### 1. Dual-Scenario Data Pipeline

Menghasilkan dua jenis dataset:

* **Empirical Dataset** (berbasis data nyata)
* **Synthetic Dataset** (dengan domain drift terkontrol)

Pendekatan ini memungkinkan evaluasi kemampuan generalisasi model terhadap perubahan distribusi data.

### 2. Dual-Layer Safety Architecture

#### Clinical Safety Bypass

Deteksi instan untuk kondisi akut:

* Suhu tubuh ≥ 40°C
* Milk Conductivity ≥ 6.5 mS/cm

#### Predictive Machine Learning Path

Menggunakan Random Forest untuk mendeteksi risiko mastitis subklinis sebelum gejala klinis muncul.

### 3. Explainable AI (XAI)

Integrasi SHAP (*SHapley Additive exPlanations*) memungkinkan:

* Identifikasi faktor fisiologis paling berpengaruh
* Transparansi proses prediksi
* Dukungan pengambilan keputusan yang dapat dijelaskan

### 4. Prescriptive Analytics Dinamis

Rekomendasi tindakan disesuaikan berdasarkan akar penyebab utama (*root cause*) seperti:

* Body Temperature
* Milk Conductivity
* Milk Yield
* Heart Rate
* Rumination

### 5. Cross-Scenario Validation Matrix

Evaluasi dilakukan pada empat kombinasi pelatihan dan pengujian:

| Training  | Testing   |
| --------- | --------- |
| Empirical | Empirical |
| Synthetic | Synthetic |
| Empirical | Synthetic |
| Synthetic | Empirical |

Tujuannya adalah mengukur ketahanan model terhadap *domain shift*.

### 6. Edge-Optimized Architecture

Dirancang untuk berjalan pada perangkat komputasi ringan dengan:

* Ukuran model sangat kecil
* Latensi inferensi rendah
* Konsumsi RAM minimal

---

## 📊 Hasil Audit dan Metrik Performa

| Parameter              | Target   | Hasil Aktual | Status |
| ---------------------- | -------- | ------------ | ------ |
| Recall (Drift 30%)     | ≥ 90%    | 97.06%       | ✅ PASS |
| F1-Score               | ≥ 85%    | 98.51%       | ✅ PASS |
| Akurasi Cross-Scenario | ≥ 90%    | 99.38%       | ✅ PASS |
| Model Size             | < 10 MB  | 0.06 MB      | ✅ PASS |
| P95 Latency            | < 50 ms  | ~ 14.9 ms       | ✅ PASS |
| Mean Latency           | < 50 ms  | ~ 14.1 ms       | ✅ PASS |
| Peak RAM               | < 100 MB | < 2 MB       | ✅ PASS |

---

## 📂 Struktur Direktori Proyek

```text
mastitis-ds-project/
├── data/
│   ├── train_empirical.csv
│   ├── test_empirical.csv
│   ├── train_synthetic.csv
│   └── test_synthetic.csv
│
├── docs/
│   └── cow_milk_mastitis_dataset.csv
│
├── models/
│   ├── mastitis_model_empirical.pkl
│   ├── mastitis_model_synthetic.pkl
│   ├── feature_importance_empirical.csv
│   ├── feature_importance_synthetic.csv
│   └── cross_validation_report.csv
│
├── src/
│   ├── data_fusion.py
│   ├── train_dual.py
│   ├── engine_dual.py
│   ├── app_dual.py
│   ├── test_suite.py
│   └── memory_audit.py
│
├── requirements.txt
└── README.md
```

---

## 🛠️ Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/adittyapratama2/mastitis-ds-project.git
cd mastitis-ds-project
```

### 2. Membuat Virtual Environment

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependency

```bash
pip install -r requirements.txt
```

---

## 📦 Tahap 1 - Generate Dataset

Pipeline menghasilkan:

* Dataset Empirical
* Dataset Synthetic
* Train/Test Split untuk masing-masing skenario

Jalankan:

```bash
python src/data_fusion.py
```

Output:

```text
data/
├── train_empirical.csv
├── test_empirical.csv
├── train_synthetic.csv
└── test_synthetic.csv
```

---

## 🤖 Tahap 2 - Training dan Validasi

Melatih dua model Random Forest:

* Model Empirical
* Model Synthetic

Sekaligus menghasilkan:

* Feature Importance
* Cross Validation Matrix
* Latency Benchmark

Jalankan:

```bash
python src/train_dual.py
```

Output:

```text
models/
├── mastitis_model_empirical.pkl
├── mastitis_model_synthetic.pkl
├── feature_importance_empirical.csv
├── feature_importance_synthetic.csv
└── cross_validation_report.csv
```

---

## 🧪 Tahap 3 - Menjalankan Test Suite

Melakukan:

* Functional Testing
* Clinical Safety Testing
* Stress Testing
* Inference Benchmarking

Jalankan:

```bash
python src/test_suite.py
```

---

## 📈 Tahap 4 - Menjalankan Dashboard

Dashboard berbasis Streamlit mendukung:

* Pemilihan model Empirical/Synthetic
* SHAP Visualization
* Risk Assessment
* Prescriptive Recommendation

Jalankan:

```bash
streamlit run src/app_dual.py
```

Kemudian buka browser:

```text
http://localhost:8501
```

---

## 🔄 Alur Sistem

```text
Sensor Data
      │
      ▼
Clinical Safety Bypass
      │
      ├── Acute Case
      │      ▼
      │   Immediate Action
      │
      ▼
Random Forest Prediction
      │
      ▼
SHAP Explainability
      │
      ▼
Root Cause Analysis
      │
      ▼
Prescriptive Recommendation
      │
      ▼
Decision Support Output
```

---

## 📋 Contoh Response Engine

Fungsi `process_inference()` menghasilkan keluaran seperti berikut:

```json
{
  "status": "SUCCESS",
  "prediction": 1,
  "probability": 0.9876,
  "confidence": 0.9876,
  "risk_tier": "HIGH",
  "action_code": "ISOLATE_AND_TEST",
  "root_cause": "Milk Conductivity",
  "root_cause_impact": 0.4321,
  "inference_time_ms": 2.34,
  "meta": {
    "urgency": "TINGGI (Segera Tindak)",
    "timeline": "Dalam waktu < 3 Jam",
    "message": "Lonjakan konduktivitas listrik terdeteksi."
  },
  "features_used": {}
}
```

---

## 🎯 Kontribusi Akademik

Penelitian ini mengusulkan pendekatan:

**Resource-Constrained Explainable Prescriptive Decision Support System for Early Mastitis Detection Using Dual-Scenario Validation on Edge Devices**

Kontribusi utama:

1. Integrasi Explainable AI (SHAP) pada perangkat Edge.
2. Prescriptive Analytics berbasis akar penyebab fisiologis.
3. Dual-Scenario Validation untuk evaluasi ketahanan domain.
4. Arsitektur ringan untuk lingkungan peternakan dengan sumber daya terbatas.

---

## 👨‍💻 Tim Pengembang

### Kelompok 3 – Data Science

| Nama               | NIM        |
| ------------------ | ---------- |
| Adittya Pratama    | 25.55.2838 |
| Tedy Nurkholis     | 25.55.2859 |
| Randi Okta Miranda | 25.55.2855 |
| Janulius           | 25.55.2848 |

---

## 📄 Lisensi

Proyek ini dikembangkan untuk kebutuhan akademik Program Magister Informatika dan penelitian sistem pendukung keputusan berbasis Data Science.

```markdown
# Resource-Constrained Early Mastitis Decision Support System (DSS) 🐄

Aplikasi Sistem Penunjang Keputusan (DSS) preskriptif berbasis *Edge Computing* untuk deteksi dini penyakit Mastitis Subklinis pada sapi perah menggunakan algoritma **Random Forest** dan **Explainable AI (SHAP)**. Sistem ini dioptimalkan khusus untuk berjalan pada perangkat komputasi lokal kandang berspesifikasi rendah (*Resource-Constrained Environment*).

---

## 📌 Fitur Utama & Keunggulan Sistem

1. **Dual-Layer Architecture:** * *Safety Alert Layer (Rule-Based):* Proteksi hulu deterministik instant bypass untuk kasus Mastitis Klinis akut.
   * *Predictive Machine Learning Path:* Model probabilistik untuk mendeteksi area abu-abu Mastitis Subklinis tersembunyi.
2. **Explainable AI (XAI) Grounded:** Integrasi lokal pustaka SHAP secara *runtime* untuk membongkar sifat *black-box* model dan mendeteksi akar masalah fisiologis utama sapi (*root cause identification*).
3. **Prescriptive Analytics Actionable Triage:** Menghasilkan rekomendasi tindakan medis lapangan spesifik dan dinamis berdasarkan kombinasi tingkat risiko penyakit dan kontribusi variabel fitur SHAP.
4. **Production-Grade Edge Footprint:** Beban memori dan latensi super hemat untuk menjamin reliabilitas *mini-PC* lokal kandang tanpa ancaman *Out-of-Memory (OOM) crash*.

---

## 📊 Hasil Audit & Metrik Performa Perangkat Lokal (Edge Node)

Berdasarkan hasil eksekusi riil dari skrip pengujian otomatis (`test_suite.py`) dan profiler memori (`memory_audit.py`) pada mesin lokal, sistem berhasil mengunci indikator performa sebagai berikut:

| Parameter Evaluasi | Target Batasan Operasional | Hasil Aktual Eksperimen | Status Kelayakan |
| :--- | :--- | :--- | :--- |
| **Sensitivitas (Recall)** | $\ge 90.00\%$ | **92.50%** (30% Domain Drift) | **PASSED** |
| **F1-Score** | $\ge 85.00\%$ | **86.55%** | **PASSED** |
| **Model Disk Size** | $< 10.0\text{ MB}$ | **0.2474 MB** (`.pkl` file) | **PASSED** |
| **Inference Latency P95** | $< 50.0\text{ ms}$ | **15.7683 ms** | **PASSED** |
| **Peak RAM Memory** | $< 100.0\text{ MB}$ | **1.7226 MB** (Tracemalloc Audit) | **PASSED** |

---

## 📂 Struktur Direktori Proyek

```text
mastitis-ds-project/
├── data/
│   ├── test_mastitis_edge.csv    # Dataset uji independen
│   └── train_mastitis_edge.csv   # Dataset latih informed-stochastic
├── models/
│   └── mastitis_model_v1.pkl     # Berkas biner model Random Forest (0.24 MB)
├── src/
│   ├── app.py                    # Frontend Dashboard Utama (Streamlit Application)
│   ├── engine.py                 # Core Bisnis Logika, Safety Layer, & SHAP Runtime
│   ├── generator.py              # Generator Stochastic Multivariat Terkorelasi
│   ├── memory_audit.py           # Utilitas Profiler RAM Tracemalloc
│   └── test_suite.py             # Automated Boundary Value & Stress Testing
├── requirements.txt              # Daftar dependensi modul python
└── README.md                     # Dokumentasi utama proyek

```

---

## 🛠️ Panduan Instalasi & Operasional Perangkat

Pastikan sistem operasi Anda sudah terpasang Python 3.11+ dan pustaka virtual environment (`venv`). Execution di terminal root folder proyek:

### 1. Inisialisasi Environment & Install Dependensi

```bash
# Membuat virtual environment terisolasi
python -m venv venv

# Aktivasi environment (Linux/macOS)
source venv/bin/activate

# Install dependensi core, frontend UI, dan XAI secara bersamaan
./venv/bin/pip install -r requirements.txt

```

*Catatan: Isi dari file `requirements.txt` minimal wajib mencakup: `scikit-learn`, `joblib`, `pandas`, `numpy`, `shap`, `matplotlib`, `streamlit`.*

### 2. Eksekusi Jalur Siklus Hidup Kode

* **Skenario Latih Model Ulang (Jika Data Berubah):**
```bash
python src/generator.py && python src/train.py

```


* **Menjalankan Automated Test Suite (Fungsional, Batas Ambang, & Latensi):**
```bash
python src/test_suite.py

```


* **Menjalankan Profiler Puncak Alokasi RAM:**
```bash
python src/memory_audit.py

```


* **Menyalakan Aplikasi Dashboard Streamlit Lokal Kandang:**
```bash
python -m streamlit run src/app.py

```



---

## 🧩 Kontrak Logika Data (API Response Specification)

Fungsi `process_reading()` pada `LivestockPrescriptiveEngine` menerima payload masukan telemetri mentah dari sensor IoT dan memprosesnya menjadi respon JSON terstandardisasi terformat di bawah ini:

```json
{
  "prediction": "Suspected Subclinical Mastitis",
  "probability": 0.9990,
  "confidence": 0.9990,
  "risk_tier": "HIGH",
  "root_cause": "Milk Conductivity",
  "root_cause_impact": 0.2196,
  "action_code": "VET_CONSULT",
  "inference_time_ms": 20.65,
  "meta": {
    "message": "Risiko tinggi subclinical mastitis. Diperlukan isolasi dan intervensi medis.\n\nTindakan Spesifik (XAI Grounded): Lakukan CMT test segera, pisahkan fluks susu terinfeksi, dan jadwalkan konsultasi veteriner.",
    "timeline": "12 jam",
    "urgency": "HIGH"
  }
}

```

---

## 👥 Kontributor Proyek Kelompok 3 Data Science

* **Adittya Pratama     - 25.55.2838** 
* **Tedy Nurkholis      - 25.55.2859** 
* **Randi Okta Miranda  - 25.55.2855** 
* **Janulius            - 25.55.2848** 

```

---

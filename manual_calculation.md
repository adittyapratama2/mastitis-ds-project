# Penjelasan Cara Kerja Manual (Bypass & XAI)
Berdasarkan hasil eksekusi dari 10 sampel dataset empiris, berikut adalah rincian penjelasan perhitungan dan proses pengambilan keputusan untuk deteksi dini mastitis menggunakan dua arsitektur yang dibuat.

## 1. Tahap Pertama: Clinical Safety Bypass
Bypass ini adalah filter cepat berbasis rules untuk kasus akut yang memerlukan penanganan seketika tanpa harus menunggu proses komputasi yang panjang dari model ML. Rule utamanya:
- **Suhu Tubuh (Temp) >= 40.0 °C**
- **Milk Conductivity (Cond) >= 6.5 mS/cm**

### Data Sampel Terbatas (10 Baris Pertama Empirical)

| ID | Temp (°C) | HR (bpm) | Rumination (m) | Milk Yield (L) | Cond (mS/cm) | Aktual | Prediksi / Rule | Keterangan |
|---|---|---|---|---|---|---|---|---|
| 1 | 39.48 | 78.3 | 542.5 | 15.2 | 4.73 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 2 | 39.96 | 85.6 | 506.3 | 18.5 | 4.56 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 3 | 38.31 | 74.2 | 563.7 | 22.9 | 4.78 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 4 | 39.36 | 67.2 | 525.3 | 18.2 | 4.25 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 5 | 38.32 | 67.9 | 488.3 | 22.2 | 4.82 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 6 | 41.50 | 94.4 | 436.0 | 12.2 | 6.50 | Sakit | Kasus Akut (Bypass) - Sakit | Suhu Tubuh Tinggi, Milk Conductivity Tinggi |
| 7 | 38.51 | 57.9 | 441.8 | 16.9 | 4.49 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 8 | 41.50 | 100.3 | 372.5 | 9.9 | 6.05 | Sakit | Kasus Akut (Bypass) - Sakit | Suhu Tubuh Tinggi |
| 9 | 39.21 | 73.8 | 558.1 | 22.4 | 4.86 | Sehat | Masuk ke Model Random Forest | Aman (Bypass) |
| 10| 41.50 | 104.1 | 430.3 | 11.5 | 6.08 | Sakit | Kasus Akut (Bypass) - Sakit | Suhu Tubuh Tinggi |

**Penjelasan Tahap 1:**
- Pada sampel **ID 6, 8, dan 10**, sistem langsung mengkategorikan sebagai **"Kasus Akut (Sakit)"** karena sapi tersebut memiliki suhu tubuh di atas 40°C. Bahkan pada sampel ID 6, Conductivity juga mencapai batas akut 6.5.
- Sampel yang lolos (status "Aman Bypass") akan dilanjutkan ke model Random Forest untuk mendeteksi risiko subklinis yang tidak terlihat oleh mata telanjang atau rules sederhana ini.

## 2. Tahap Kedua: Fitur Explaination (SHAP) pada Random Forest
Bagi data yang lolos Clinical Safety Bypass (kasus subklinis), model Random Forest akan melakukan klasifikasi. Untuk mengetahui alasan (why) mengapa model memutuskan sehat atau sakit, digunakan SHAP (SHapley Additive exPlanations).

Nilai **Base Value (Expected Value)** model kami adalah sekitar `0.50` (peluang mastitis). Nilai fitur (SHAP value) bernilai positif (+) berarti fitur tersebut **mendorong probabilitas menjadi sakit**, sedangkan nilai negatif (-) berarti fitur tersebut **mendorong probabilitas menjadi sehat**.

Berikut rincian dari sampel ID 1 (Sapi Sehat) dan ID 6 (Sapi Sakit) berdasarkan SHAP values:

### Analisis Sampel Sehat (ID 1)
Data Aktual: `Temp: 39.48, HR: 78.3, Ruminasi: 542.5, Yield: 15.2, Cond: 4.73`
- **milk_conductivity**: `-0.2856` (Menurunkan risiko mastitis paling kuat)
- **body_temperature**: `-0.1286` (Menurunkan risiko mastitis)
- **milk_yield**: `-0.0567` (Menurunkan risiko mastitis)
- **heart_rate**: `-0.0259` (Menurunkan risiko mastitis)
- **rumination_time**: `-0.0034` (Menurunkan risiko mastitis)
- **Kesimpulan**: Conductivity (4.73) dan Temperature (39.48) berada di ambang normal yang aman. Semuanya berkontribusi negatif (-) sehingga total probabilitas berada jauh di bawah base value, model yakin sapi dalam keadaan **SEHAT**.

### Analisis Sampel Sakit Subklinis/Akut (ID 6)
Data Aktual: `Temp: 41.5, HR: 94.4, Ruminasi: 436.0, Yield: 12.2, Cond: 6.5`
- **milk_conductivity**: `+0.2787` (Meningkatkan risiko mastitis paling kuat)
- **body_temperature**: `+0.1371` (Meningkatkan risiko mastitis)
- **milk_yield**: `+0.0600` (Meningkatkan risiko mastitis karena produksi susut tajam)
- **heart_rate**: `+0.0238` (Meningkatkan risiko mastitis karena denyut jantung di atas normal)
- **rumination_time**: `+0.0003` (Meningkatkan risiko mastitis tipis)
- **Kesimpulan**: Walaupun misalnya sapi ini lolos bypass (misal Cond 6.0 dan Temp 39.9, contoh ID lain), lonjakan Conductivity dan Temperature ini mendominasi peningkatan probabilitas (+0.27 dan +0.13) yang akan melebihi treshold `0.5`, sehingga model mendiagnosis sapi **SAKIT**. SHAP memberikan kejelasan bahwa akar penyebab utama di ID 6 adalah Conductivity dan Temperature yang terlalu tinggi.

Dengan metode XAI (SHAP) ini, kelompok kami tidak hanya menghasilkan klasifikasi "Sakit" atau "Sehat" (Black Box), melainkan mampu memberikan penjelasan akar masalah secara transparan kepada peternak sehingga rekomendasi penanganan yang diberikan bisa jauh lebih akurat (White Box).

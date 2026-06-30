import pandas as pd
import numpy as np

# Ambil 10 data teratas dari empirical data
df = pd.read_csv('data/train_empirical.csv').head(10)

def clinical_safety_bypass(row):
    # Rule dari README
    # Suhu tubuh >= 40°C
    # Milk Conductivity >= 6.5 mS/cm
    
    is_acute = False
    reason = []
    
    if row['body_temperature'] >= 40.0:
        is_acute = True
        reason.append('Suhu Tubuh Tinggi')
    
    if row['milk_conductivity'] >= 6.5:
        is_acute = True
        reason.append('Milk Conductivity Tinggi')
        
    return is_acute, reason

results = []
for idx, row in df.iterrows():
    is_acute, reason = clinical_safety_bypass(row)
    status_aktual = "Sakit" if row['mastitis_status'] == 1 else "Sehat"
    
    if is_acute:
        prediksi = "Kasus Akut (Bypass) - Sakit"
        alasan = ", ".join(reason)
    else:
        prediksi = "Masuk ke Model Random Forest"
        alasan = "Aman (Bypass)"
        
    results.append({
        'ID': idx + 1,
        'Temp': row['body_temperature'],
        'HR': round(row['heart_rate'], 1),
        'Rumination': round(row['rumination_time'], 1),
        'Milk Yield': row['milk_yield'],
        'Cond': row['milk_conductivity'],
        'Aktual': status_aktual,
        'Prediksi/Rule': prediksi,
        'Keterangan': alasan
    })

df_res = pd.DataFrame(results)
print("=== PERHITUNGAN MANUAL (Clinical Safety Bypass & Random Forest Path) ===")
print(df_res.to_markdown(index=False))

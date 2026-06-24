# src/generator.py
import pandas as pd
import numpy as np
import os

class LivestockDataGenerator:
    # Operational Simulation Thresholds (Critical Alert Layer)
    CRITICAL_TEMP_THRES = 40.0
    CRITICAL_COND_THRES = 6.5
    
    # Engineering Approximations untuk korelasi biologis literatur
    COEFF_HEART_RATE = 12
    COEFF_RUMINATION = 140
    COEFF_MILK_YIELD = 2.5

    def __init__(self, seed=42):
        self.seed = seed
        self.baseline = {
            'body_temperature': 38.5, 'milk_conductivity': 4.8,
            'heart_rate': 70, 'rumination_time': 480, 'milk_yield': 25.0
        }

    def generate_shifted_data(self, num_records=1200, mastitis_ratio=0.20, drift_factor=1.0):
        rand = np.random.RandomState(self.seed)
        n_healthy = int(num_records * (1 - mastitis_ratio))
        n_subclinical = num_records - n_healthy
        
        # 1. Base Independent Features (Suhu & Konduktivitas)
        bt_h = rand.normal(loc=38.5, scale=0.15 * drift_factor, size=n_healthy)
        mc_h = rand.normal(loc=4.8, scale=0.20 * drift_factor, size=n_healthy)
        
        bt_s = rand.normal(loc=38.9, scale=0.25 * drift_factor, size=n_subclinical)
        mc_s = rand.normal(loc=5.6, scale=0.35 * drift_factor, size=n_subclinical)
        
        df_h = pd.DataFrame({'body_temperature': bt_h, 'milk_conductivity': mc_h, 'label': 0})
        df_s = pd.DataFrame({'body_temperature': bt_s, 'milk_conductivity': mc_s, 'label': 1})
        df = pd.concat([df_h, df_s]).sample(frac=1, random_state=rand).reset_index(drop=True)
        
        # 2. Inject Biological Dependencies (Correlated Multivariate)
        df['heart_rate'] = (
            70 + (df['body_temperature'] - self.baseline['body_temperature']) * self.COEFF_HEART_RATE 
            + rand.normal(0, 4 * drift_factor, size=len(df))
        ).astype(int)
        
        df['rumination_time'] = (
            480 - (df['body_temperature'] - self.baseline['body_temperature']) * self.COEFF_RUMINATION 
            + rand.normal(0, 20 * drift_factor, size=len(df))
        ).astype(int)
        
        df['milk_yield'] = (
            25.0 - (df['milk_conductivity'] - self.baseline['milk_conductivity']) * self.COEFF_MILK_YIELD 
            + rand.normal(0, 1.5 * drift_factor, size=len(df))
        )
        
        # 3. Post-Processing & Sensor Noise Simulation
        df['body_temperature'] += rand.normal(0, 0.05 * drift_factor, size=len(df))
        df['milk_conductivity'] += rand.normal(0, 0.08 * drift_factor, size=len(df))
        
        # Clip data berdasarkan batas kritis operasional darurat
        df['body_temperature'] = df['body_temperature'].clip(37.0, self.CRITICAL_TEMP_THRES)
        df['milk_conductivity'] = df['milk_conductivity'].clip(4.0, self.CRITICAL_COND_THRES)
        df['rumination_time'] = df['rumination_time'].clip(100, 600)
        
        df.insert(0, 'cow_id', [f"COW-2026-{rand.randint(1000, 9999)}" for _ in range(len(df))])
        return df

if __name__ == "__main__":
    # Tentukan path relatif terhadap lokasi script src/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), 'data')
    
    # Buat folder data jika belum ada
    os.makedirs(data_dir, exist_ok=True)
    
    # Produksi Dataset Latih (Clean Reference)
    train_gen = LivestockDataGenerator(seed=42)
    df_train = train_gen.generate_shifted_data(num_records=1000, drift_factor=1.0)
    df_train.to_csv(os.path.join(data_dir, 'train_mastitis_edge.csv'), index=False)
    
    # Produksi Dataset Uji (Dirty Test Set - 30% Variance Drift)
    test_gen = LivestockDataGenerator(seed=99)
    df_test = test_gen.generate_shifted_data(num_records=400, drift_factor=1.3)
    df_test.to_csv(os.path.join(data_dir, 'test_mastitis_edge.csv'), index=False)
    
    print("[SUCCESS] Langkah 1 Selesai. Workspace data terisi penuh:")
    print(f" -> data/train_mastitis_edge.csv ({len(df_train)} records)")
    print(f" -> data/test_mastitis_edge.csv ({len(df_test)} records dengan domain shift)")
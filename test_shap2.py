import pandas as pd
import joblib
import numpy as np
import shap

# Load model and data
model = joblib.load('models/mastitis_model_empirical.pkl')
df = pd.read_csv('data/train_empirical.csv').head(10)
X = df.drop(columns=['mastitis_status'])

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("SHAP shape:", np.array(shap_values).shape)

# Handle different shap_values formats returned by TreeExplainer
if isinstance(shap_values, list):
    sv = shap_values[1]  # class 1 (mastitis)
elif len(np.array(shap_values).shape) == 3:
    sv = shap_values[:, :, 1] # shape (samples, features, classes)
else:
    sv = shap_values

for i in range(10):
    print(f"\n--- Sampel ID {i+1} ---")
    print(f"Data: {X.iloc[i].to_dict()}")
    
    # Zip features with their shap values
    contributions = list(zip(X.columns, sv[i]))
    sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    
    print("Kontribusi Fitur (SHAP):")
    for feat, val in sorted_contributions:
        impact = "Meningkatkan" if val > 0 else "Menurunkan"
        print(f"  - {feat}: {val:.4f} ({impact} risiko mastitis)")

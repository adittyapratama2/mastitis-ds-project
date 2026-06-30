import pandas as pd
import joblib
import numpy as np

# Load model and data
model = joblib.load('models/mastitis_model_empirical.pkl')
df = pd.read_csv('data/train_empirical.csv').head(10)
X = df.drop(columns=['mastitis_status'])

try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    print("SHAP Values shape:", np.array(shap_values).shape)
    print("Expected value:", explainer.expected_value)
    
    # Random Forest in scikit-learn usually returns shap values for each class
    if isinstance(shap_values, list):
        # We want the explanation for class 1 (Mastitis)
        sv = shap_values[1]
    else:
        sv = shap_values
        
    for i in range(10):
        print(f"\n--- Sampel ID {i+1} ---")
        print(f"Data: {X.iloc[i].to_dict()}")
        
        # Zip features with their shap values
        contributions = zip(X.columns, sv[i])
        sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
        
        print("Kontribusi Fitur (SHAP):")
        for feat, val in sorted_contributions:
            impact = "Meningkatkan" if val > 0 else "Menurunkan"
            print(f"  - {feat}: {val:.4f} ({impact} risiko mastitis)")
        
except ImportError:
    print("SHAP not installed")

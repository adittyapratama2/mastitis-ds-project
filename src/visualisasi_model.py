import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Gunakan backend Agg untuk memastikan tidak ada warning
import matplotlib
matplotlib.use('Agg')  # <- Tambahkan ini di awal

feature_names = ['body_temperature', 'heart_rate', 'rumination_time', 'milk_yield', 'milk_conductivity']
class_names = ['Healthy', 'Subclinical']

model_emp = joblib.load('models/mastitis_model_empirical.pkl')
model_syn = joblib.load('models/mastitis_model_synthetic.pkl')

fig, axes = plt.subplots(1, 2, figsize=(24, 10), dpi=300)

plot_tree(model_emp.estimators_[0],
          feature_names=feature_names,
          class_names=class_names,
          max_depth=2,
          filled=True,
          ax=axes[0])
axes[0].set_title('Model Empiris - Pohon 1', fontsize=14, fontweight='bold')

plot_tree(model_syn.estimators_[0],
          feature_names=feature_names,
          class_names=class_names,
          max_depth=2,
          filled=True,
          ax=axes[1])
axes[1].set_title('Model Sintetis - Pohon 1', fontsize=14, fontweight='bold')

plt.tight_layout()

# Simpan ke file (tanpa menampilkan)
plt.savefig('models/decision_tree_comparison.png', bbox_inches='tight', dpi=300)
print("Gambar berhasil disimpan ke models/decision_tree_comparison.png")
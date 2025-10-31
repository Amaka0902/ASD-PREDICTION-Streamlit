# ============================================
#  Add Label Noise to ASD Dataset
#  Author: Amaka (Pre-Diagnosis ASD Project)
# ============================================

import pandas as pd
import numpy as np

# -------------------------------
# 1️⃣ Load Original Dataset
# -------------------------------
data = pd.read_csv("train_cleaned.csv")

# -------------------------------
# 2️⃣ Define Noise Parameters
# -------------------------------
target_col = "Class/ASD"   # Target variable
flip_rate = 0.10            # 10% label flips (you can adjust to 0.05 or 0.15)

# -------------------------------
# 3️⃣ Apply Label Noise
# -------------------------------
# Make a copy to avoid overwriting original
noisy_data = data.copy()

# Randomly select indices to flip
n_samples = len(noisy_data)
flip_indices = np.random.choice(n_samples, size=int(flip_rate * n_samples), replace=False)

# Flip binary labels: 0 -> 1 and 1 -> 0
noisy_data.loc[flip_indices, target_col] = 1 - noisy_data.loc[flip_indices, target_col]

# -------------------------------
# 4️⃣ Save New File
# -------------------------------
noisy_data.to_csv("train_label_noise.csv", index=False)

# -------------------------------
# 5️⃣ Print Summary
# -------------------------------
print(f"✅ Label noise added successfully!")
print(f"   → {len(flip_indices)} out of {n_samples} labels flipped ({flip_rate*100:.0f}%)")
print("   Saved as: train_label_noise.csv")

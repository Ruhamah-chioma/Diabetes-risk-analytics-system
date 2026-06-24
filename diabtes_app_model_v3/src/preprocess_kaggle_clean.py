import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

# Paths
dataset_path = r"C:\Users\Ruhamah\Desktop\Prediabetes app\prediabtes_app_model_v3\dataset\diabetes_prediction_dataset.csv"
processed_dir = r"C:\Users\Ruhamah\Desktop\Prediabetes app\prediabtes_app_model_v3\dataset\processed"

print("="*60)
print("CLEAN KAGGLE PREPROCESSING")
print("="*60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"   Shape: {df.shape}")

# Encode categorical variables
print("\n2. Encoding categorical variables...")
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

smoking_map = {
    'never': 0, 'former': 1, 'current': 2,
    'not current': 1, 'ever': 1, 'No Info': 0
}
df['smoking_history'] = df['smoking_history'].map(smoking_map)

# Select features
feature_cols = ['gender', 'age', 'hypertension', 'heart_disease', 
                'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']

X = df[feature_cols].copy()
y = df['diabetes'].copy()

print(f"   Features: {feature_cols}")
print(f"   Target distribution:\n   {y.value_counts().to_dict()}")

# Remove duplicates
print("\n3. Removing duplicates...")
initial = len(X)
X = X.drop_duplicates()
y = y.loc[X.index]
print(f"   Removed {initial - len(X)} duplicates")

# Handle missing values - THE KEY FIX
print("\n4. Handling missing values...")
print(f"   Missing before: {X.isnull().sum().sum()}")

# Use SimpleImputer
imputer = SimpleImputer(strategy='median')
X_array = imputer.fit_transform(X)
print(f"   Missing after imputation: {np.isnan(X_array).sum()}")

# Convert back to DataFrame for easier handling
X = pd.DataFrame(X_array, columns=feature_cols)

# Split data
print("\n5. Splitting data...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

val_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
)

print(f"   Training: {len(X_train)}")
print(f"   Validation: {len(X_val)}")
print(f"   Test: {len(X_test)}")

# Normalize
print("\n6. Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Final NaN check
print("\n7. Final NaN check:")
print(f"   X_train has NaN: {np.isnan(X_train_scaled).any()}")
print(f"   X_val has NaN: {np.isnan(X_val_scaled).any()}")
print(f"   X_test has NaN: {np.isnan(X_test_scaled).any()}")
print(f"   y_train has NaN: {np.isnan(y_train.values).any()}")

# Save
print("\n8. Saving processed data...")
os.makedirs(processed_dir, exist_ok=True)

np.save(os.path.join(processed_dir, "X_train.npy"), X_train_scaled.astype(np.float32))
np.save(os.path.join(processed_dir, "X_val.npy"), X_val_scaled.astype(np.float32))
np.save(os.path.join(processed_dir, "X_test.npy"), X_test_scaled.astype(np.float32))
np.save(os.path.join(processed_dir, "y_train.npy"), y_train.values.astype(np.int32))
np.save(os.path.join(processed_dir, "y_val.npy"), y_val.values.astype(np.int32))
np.save(os.path.join(processed_dir, "y_test.npy"), y_test.values.astype(np.int32))

joblib.dump(feature_cols, os.path.join(processed_dir, "feature_names.pkl"))
joblib.dump(scaler, os.path.join(processed_dir, "scaler.pkl"))
joblib.dump(imputer, os.path.join(processed_dir, "imputer.pkl"))

print(f"   ✓ Data saved to {processed_dir}")

print("\n" + "="*60)
print("✅ PREPROCESSING COMPLETE!")
print("="*60)
print(f"\n📊 SUMMARY:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(feature_cols)}")
print(f"   Diabetes rate: {y.mean()*100:.1f}%")
print("="*60)
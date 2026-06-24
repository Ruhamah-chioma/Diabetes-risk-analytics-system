import pandas as pd
import os

# Path to your Kaggle dataset
dataset_path = r"C:\Users\Ruhamah\Desktop\Prediabetes app\prediabtes_app_model_v3_nov2\dataset\diabetes_prediction_dataset.csv"

print("="*50)
print("KAGGLE DATASET EXPLORATION")
print("="*50)

# Load data
df = pd.read_csv(dataset_path)

print(f"\n📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\n📋 Column Names:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n🔍 First 5 rows:")
print(df.head())

print(f"\n📈 Target Distribution (diabetes):")
print(df['diabetes'].value_counts())
print(f"   Percentage with diabetes: {df['diabetes'].mean()*100:.1f}%")

print(f"\n📊 Data Types:")
print(df.dtypes)

print(f"\n❓ Missing Values:")
print(df.isnull().sum())

print(f"\n📊 Basic Statistics:")
print(df.describe())
import pandas as pd
import os

# Change this to where your NHANES file is
file_path = r"nhanes.csv"  # UPDATE THIS

print("="*60)
print("NHANES DATA EXPLORER")
print("="*60)

# Load the file
df = pd.read_csv(file_path)

print(f"\nFile shape: {df.shape}")
print(f"\nFirst 5 column names:")
for i, col in enumerate(df.columns[:20]):
    print(f"   {i+1}. {col}")

print(f"\nFirst 3 rows of data:")
print(df.head(3))

print(f"\nBasic statistics for numeric columns:")
print(df.describe())
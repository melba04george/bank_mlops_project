import pandas as pd

# Load the CSV with correct separator
df = pd.read_csv("data/bank.csv", sep=';')

# Show dataset shape
print(f"Dataset shape: {df.shape}")

# Show first few rows
print("Sample rows:")
print(df.head())

# Check target value distribution
print("\nTarget variable distribution:")
print(df['y'].value_counts())

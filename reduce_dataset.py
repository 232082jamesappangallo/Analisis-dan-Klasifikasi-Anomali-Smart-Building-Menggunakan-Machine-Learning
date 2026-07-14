import pandas as pd
import numpy as np

# Set random seed
np.random.seed(42)

print("Reducing dataset size...")

# Load original data
df_normal_train = pd.read_csv('data/normal_training.csv')
df_faulty_train = pd.read_csv('data/faulty_training.csv')
df_normal_test = pd.read_csv('data/normal_testing.csv')
df_faulty_test = pd.read_csv('data/faulty_testing.csv')

print(f"Original sizes:")
print(f"  normal_training: {len(df_normal_train)}")
print(f"  faulty_training: {len(df_faulty_train)}")
print(f"  normal_testing: {len(df_normal_test)}")
print(f"  faulty_testing: {len(df_faulty_test)}")

# Reduce to 10% of original size (minimum 100/50 samples)
sample_size_train_normal = max(100, len(df_normal_train) // 10)
sample_size_train_faulty = max(100, len(df_faulty_train) // 10)
sample_size_test_normal = max(50, len(df_normal_test) // 10)
sample_size_test_faulty = max(50, len(df_faulty_test) // 10)

# Sample
df_normal_train_reduced = df_normal_train.sample(n=sample_size_train_normal, random_state=42)
df_faulty_train_reduced = df_faulty_train.sample(n=sample_size_train_faulty, random_state=42)
df_normal_test_reduced = df_normal_test.sample(n=sample_size_test_normal, random_state=42)
df_faulty_test_reduced = df_faulty_test.sample(n=sample_size_test_faulty, random_state=42)

# Save reduced versions (overwrite original)
df_normal_train_reduced.to_csv('data/normal_training.csv', index=False)
df_faulty_train_reduced.to_csv('data/faulty_training.csv', index=False)
df_normal_test_reduced.to_csv('data/normal_testing.csv', index=False)
df_faulty_test_reduced.to_csv('data/faulty_testing.csv', index=False)

print(f"\nReduced sizes (saved):")
print(f"  normal_training: {len(df_normal_train_reduced)}")
print(f"  faulty_training: {len(df_faulty_train_reduced)}")
print(f"  normal_testing: {len(df_normal_test_reduced)}")
print(f"  faulty_testing: {len(df_faulty_test_reduced)}")

total_original = len(df_normal_train) + len(df_faulty_train) + len(df_normal_test) + len(df_faulty_test)
total_reduced = len(df_normal_train_reduced) + len(df_faulty_train_reduced) + len(df_normal_test_reduced) + len(df_faulty_test_reduced)
reduction = ((total_original - total_reduced) / total_original) * 100

print(f"\nTotal: {total_original:,} → {total_reduced:,} samples")
print(f"Reduction: {reduction:.1f}%")
print(f"\n✅ Dataset size reduced successfully!")

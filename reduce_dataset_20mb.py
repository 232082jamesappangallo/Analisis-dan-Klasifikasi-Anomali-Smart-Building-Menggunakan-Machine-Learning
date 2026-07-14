import pandas as pd
import numpy as np
import os

# Set random seed
np.random.seed(42)

print("Reducing dataset to 20 MB target...")
print()

# Load current data
df_normal_train = pd.read_csv('data/normal_training.csv')
df_faulty_train = pd.read_csv('data/faulty_training.csv')
df_normal_test = pd.read_csv('data/normal_testing.csv')
df_faulty_test = pd.read_csv('data/faulty_testing.csv')

print(f"Current sizes:")
print(f"  normal_training.csv: {len(df_normal_train):,} rows")
print(f"  faulty_training.csv: {len(df_faulty_train):,} rows")
print(f"  normal_testing.csv: {len(df_normal_test):,} rows")
print(f"  faulty_testing.csv: {len(df_faulty_test):,} rows")
total_rows = len(df_normal_train) + len(df_faulty_train) + len(df_normal_test) + len(df_faulty_test)
print(f"  Total: {total_rows:,} rows")

# Calculate current file sizes
current_size = sum(os.path.getsize(f'data/{fname}') for fname in 
                   ['normal_training.csv', 'faulty_training.csv', 'normal_testing.csv', 'faulty_testing.csv'])
current_size_mb = current_size / (1024 * 1024)
print(f"  Current size: {current_size_mb:.2f} MB")
print()

# Target: 20 MB, we want to be safe so aim for ~18 MB
target_mb = 18
reduction_factor = target_mb / current_size_mb

print(f"Target size: {target_mb} MB (from {current_size_mb:.2f} MB)")
print(f"Reduction factor: {reduction_factor:.4f} ({reduction_factor*100:.2f}%)")
print()

# Calculate new sample sizes (maintain class ratios)
new_train_normal = max(50, int(len(df_normal_train) * reduction_factor))
new_train_faulty = max(50, int(len(df_faulty_train) * reduction_factor))
new_test_normal = max(25, int(len(df_normal_test) * reduction_factor))
new_test_faulty = max(25, int(len(df_faulty_test) * reduction_factor))

print(f"New target sizes:")
print(f"  normal_training: {new_train_normal:,} rows")
print(f"  faulty_training: {new_train_faulty:,} rows")
print(f"  normal_testing: {new_test_normal:,} rows")
print(f"  faulty_testing: {new_test_faulty:,} rows")
print()

# Sample and reduce
df_normal_train_reduced = df_normal_train.sample(n=new_train_normal, random_state=42)
df_faulty_train_reduced = df_faulty_train.sample(n=new_train_faulty, random_state=42)
df_normal_test_reduced = df_normal_test.sample(n=new_test_normal, random_state=42)
df_faulty_test_reduced = df_faulty_test.sample(n=new_test_faulty, random_state=42)

# Save reduced versions
df_normal_train_reduced.to_csv('data/normal_training.csv', index=False)
df_faulty_train_reduced.to_csv('data/faulty_training.csv', index=False)
df_normal_test_reduced.to_csv('data/normal_testing.csv', index=False)
df_faulty_test_reduced.to_csv('data/faulty_testing.csv', index=False)

print("✅ Files saved")
print()

# Check new sizes
new_size = sum(os.path.getsize(f'data/{fname}') for fname in 
               ['normal_training.csv', 'faulty_training.csv', 'normal_testing.csv', 'faulty_testing.csv'])
new_size_mb = new_size / (1024 * 1024)

print(f"New file sizes:")
print(f"  normal_training.csv: {os.path.getsize('data/normal_training.csv')/(1024*1024):.2f} MB")
print(f"  faulty_training.csv: {os.path.getsize('data/faulty_training.csv')/(1024*1024):.2f} MB")
print(f"  normal_testing.csv: {os.path.getsize('data/normal_testing.csv')/(1024*1024):.2f} MB")
print(f"  faulty_testing.csv: {os.path.getsize('data/faulty_testing.csv')/(1024*1024):.2f} MB")
print(f"  Total: {new_size_mb:.2f} MB")
print()

total_rows_new = len(df_normal_train_reduced) + len(df_faulty_train_reduced) + len(df_normal_test_reduced) + len(df_faulty_test_reduced)
reduction_pct = ((total_rows - total_rows_new) / total_rows) * 100

print(f"Summary:")
print(f"  Rows: {total_rows:,} → {total_rows_new:,} ({reduction_pct:.1f}% reduction)")
print(f"  Size: {current_size_mb:.2f} MB → {new_size_mb:.2f} MB ({(1-new_size_mb/current_size_mb)*100:.1f}% reduction)")
print()
print(f"✅ Dataset successfully reduced to ~20 MB!")

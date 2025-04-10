import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the analysis log file
print("Loading ECG analysis log file...")
try:
    df = pd.read_csv('ecg_analysis_log.csv')
    print(f"Successfully loaded data with {len(df)} records")
except Exception as e:
    print(f"Error loading file: {str(e)}")
    exit(1)

# Display basic information about the dataset
print("\nDataset Overview:")
print(f"Number of unique files: {df['file_hash'].nunique()}")
print(f"Columns: {', '.join(df.columns)}")

# Check for consistency within same files
print("\nChecking consistency within same files...")
for file_hash in df['file_hash'].unique():
    file_data = df[df['file_hash'] == file_hash]
    if len(file_data) > 1:
        print(f"\nFile {file_hash} ({file_data['filename'].iloc[0]}):")
        print(f"  Records: {len(file_data)}")
        print(f"  AF probability range: {file_data['af_probability'].min():.4f} - {file_data['af_probability'].max():.4f}")
        print(f"  AF probability std dev: {file_data['af_probability'].std():.6f}")
        print(f"  Heart rate range: {file_data['heart_rate'].min():.2f} - {file_data['heart_rate'].max():.2f} BPM")

# Check for uniqueness across different files
print("\nChecking uniqueness across different files...")
file_summaries = df.groupby('file_hash').agg({
    'filename': 'first',
    'af_probability': ['mean', 'std'],
    'heart_rate': ['mean', 'std'],
    'rmssd': ['mean', 'std'],
    'irregularity': ['mean', 'std'],
    'signal_mean': ['mean', 'std'],
    'signal_std': ['mean', 'std']
}).reset_index()

# Print key metrics for each file
print("\nKey metrics by file:")
for _, row in file_summaries.iterrows():
    print(f"\nFile: {row['filename']['first']} ({row['file_hash']})")
    print(f"  AF Probability: {row['af_probability']['mean']:.4f} ± {row['af_probability']['std']:.4f}")
    print(f"  Heart Rate: {row['heart_rate']['mean']:.2f} ± {row['heart_rate']['std']:.2f} BPM")
    print(f"  RMSSD: {row['rmssd']['mean']:.4f} ± {row['rmssd']['std']:.4f}")
    print(f"  Irregularity: {row['irregularity']['mean']:.4f} ± {row['irregularity']['std']:.4f}")

# Create visualizations
print("\nCreating visualizations...")

# 1. AF probability by file
plt.figure(figsize=(10, 6))
sns.boxplot(x='file_hash', y='af_probability', data=df)
plt.title('AF Probability by File')
plt.xlabel('File Hash')
plt.ylabel('AF Probability')
plt.savefig('af_probability_by_file.png')

# 2. Create correlation heatmap
plt.figure(figsize=(12, 8))
numeric_cols = ['af_probability', 'heart_rate', 'rmssd', 'pnn50', 'irregularity', 'signal_mean', 'signal_std', 'signal_range']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Metrics')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# 3. Compare key metrics across files
plt.figure(figsize=(12, 8))
metrics = ['af_probability', 'heart_rate', 'irregularity', 'rmssd']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    sns.boxplot(x='file_hash', y=metric, data=df, ax=axes[i])
    axes[i].set_title(f'{metric} by File')
    axes[i].set_xlabel('File Hash')
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig('metrics_comparison.png')

print("\nAnalysis complete. Visualizations saved to current directory.")
print("Results indicate whether different files produce unique analysis results.") 
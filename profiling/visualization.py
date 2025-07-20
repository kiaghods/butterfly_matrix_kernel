"""
visualization for profiling results from torch profiler
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# Look for the SQLite file in the results/profiles directory
sqlite_path = 'results/profiles/butterfly_profile.sqlite'
if not os.path.exists(sqlite_path):
    # Fallback to root directory
    sqlite_path = 'butterfly_profile.sqlite'
    if not os.path.exists(sqlite_path):
        print(f"Error: Could not find butterfly_profile.sqlite in results/profiles/ or root directory")
        exit(1)

print(f"Using SQLite file: {sqlite_path}")

# Ensure plots directory exists
os.makedirs('results/plots', exist_ok=True)

# Connect to the database
conn = sqlite3.connect(sqlite_path)

# Get kernel data with proper name resolution
kernels_df = pd.read_sql_query("""
    SELECT 
        k.start,
        k.end,
        (k.end - k.start) / 1000.0 as duration_ms,
        s.value as kernel_name,
        k.gridX, k.gridY, k.gridZ,
        k.blockX, k.blockY, k.blockZ
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.shortName = s.id
    ORDER BY k.start
""", conn)

print("Kernel data:")
print(kernels_df)

# Create timeline plot
plt.figure(figsize=(15, 8))

# Plot each kernel as a horizontal bar
for i, row in kernels_df.iterrows():
    start_ms = row['start'] / 1e6  # Convert nanoseconds to milliseconds
    duration_ms = row['duration_ms']
    
    plt.barh(i, duration_ms, left=start_ms, 
             label=row['kernel_name'] if i == 0 else "",
             alpha=0.7)

plt.xlabel('Time (ms)')
plt.ylabel('Kernel Launch')
plt.title('GPU Kernel Timeline')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/kernel_timeline.png', dpi=150, bbox_inches='tight')
plt.show()

# Create duration distribution plot
plt.figure(figsize=(10, 6))
plt.hist(kernels_df['duration_ms'], bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Kernel Duration (ms)')
plt.ylabel('Frequency')
plt.title('Kernel Duration Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/kernel_duration_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nKernel Summary Statistics:")
print(f"Total kernels: {len(kernels_df)}")
print(f"Average duration: {kernels_df['duration_ms'].mean():.2f} ms")
print(f"Min duration: {kernels_df['duration_ms'].min():.2f} ms")
print(f"Max duration: {kernels_df['duration_ms'].max():.2f} ms")
print(f"Std deviation: {kernels_df['duration_ms'].std():.2f} ms")

# Show grid/block configurations
print("\nGrid/Block Configurations:")
configs = kernels_df.groupby(['gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ']).size()
print(configs)

conn.close()
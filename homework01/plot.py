#hello
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all your CSV files
core_data = {
    8: pd.read_csv('csv/8error.csv'),
    16: pd.read_csv('csv/16error.csv'), 
    32: pd.read_csv('csv/32error.csv'),
    64: pd.read_csv('csv/64error.csv'),
    128: pd.read_csv('csv/128error.csv')
}

# Define colors with opacity gradient
colors = {
    8: (0.1, 0.1, 0.8, 0.3),    # Blue, low opacity
    16: (0.3, 0.3, 0.9, 0.5),
    32: (0.5, 0.5, 1.0, 0.7), 
    64: (0.7, 0.2, 0.7, 0.8),
    128: (0.9, 0.1, 0.1, 1.0)   # Red, full opacity
}

# PLOT 1: Speedup vs Steps (all cores)
plt.figure(figsize=(8, 6))
for cores, df in core_data.items():
    color = colors[cores]
    plt.plot(df['steps'], df['speedup_integration'], 'o-', 
             color=color, label=f'{cores} cores', linewidth=2, alpha=color[3])
plt.xlabel('Number of Steps')
plt.ylabel('Speedup (Serial/Parallel)')
plt.title('Speedup vs Problem Size (All Core Counts)')
plt.xscale('log')
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speedup_all_cores.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 2: Parallel Time vs Steps (all cores)
plt.figure(figsize=(8, 6))
for cores, df in core_data.items():
    color = colors[cores]
    plt.loglog(df['steps'], df['parallel_integration_time'], 's-',
               color=color, label=f'{cores} cores', linewidth=2, alpha=color[3])
plt.xlabel('Number of Steps')
plt.ylabel('Time (seconds)')
plt.title('Parallel Time vs Problem Size (All Core Counts)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig('parallel_time_all_cores.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 3: Error vs Time (all cores)
plt.figure(figsize=(8, 6))
for cores, df in core_data.items():
    color = colors[cores]
    plt.loglog(df['integration_error'], df['parallel_integration_time'], 'o',
               color=color, label=f'{cores} cores', alpha=color[3], markersize=8)
plt.xlabel('Absolute Error from π')
plt.ylabel('Time (seconds)')
plt.title('Time vs Accuracy Trade-off (All Core Counts)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig('error_time_all_cores.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 4: Efficiency vs Steps (all cores)
plt.figure(figsize=(8, 6))
for cores, df in core_data.items():
    color = colors[cores]
    efficiency = df['speedup_integration'] / cores
    plt.plot(df['steps'], efficiency, '^-', 
             color=color, label=f'{cores} cores', linewidth=2, alpha=color[3])
plt.xlabel('Number of Steps')
plt.ylabel('Efficiency (Speedup / Cores)')
plt.title('Parallel Efficiency vs Problem Size')
plt.xscale('log')
plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Ideal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('efficiency_all_cores.png', dpi=300, bbox_inches='tight')
plt.close()
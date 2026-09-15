
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('results/debate_rounds_experiment.csv')

# Calculate aggregation
summary = df.groupby('rounds').agg({
    'is_blocked': 'mean',
    'duration': 'mean',
    'tokens': 'mean'
}).reset_index()

# Rename for display
summary['Interception Rate %'] = summary['is_blocked'] * 100
summary['Latency (s)'] = summary['duration']

# Set style
sns.set_theme(style="white", rc={"axes.grid": False})
plt.rcParams['font.family'] = 'DejaVu Sans'

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Interception Rate
# Define colors: Highlight the winner (Round 2)
colors = ['#b0c4de' if r != 2 else '#4682b4' for r in summary['rounds']]

bars = sns.barplot(x='rounds', y='Interception Rate %', data=summary, ax=ax1, palette=colors)
ax1.set_title('Safety Performance by Debate Depth', fontsize=12, fontweight='bold', pad=15)
ax1.set_xlabel('Debate Rounds', fontsize=10)
ax1.set_ylabel('Interception Rate (%)', fontsize=10)
ax1.set_ylim(0, 100)

# Add values on top of bars
for i, v in enumerate(summary['Interception Rate %']):
    ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Add annotation for Round 2
ax1.annotate('Optimal Safety', 
             xy=(1, summary.loc[1, 'Interception Rate %']), 
             xytext=(1, summary.loc[1, 'Interception Rate %'] + 10),
             ha='center',
             arrowprops=dict(arrowstyle="->", color='black'))

# Plot 2: Latency
sns.lineplot(x='rounds', y='Latency (s)', data=summary, ax=ax2, marker='o', markersize=10, linewidth=2, color='#cd5c5c')
ax2.set_title('Cost Analysis: Latency vs rounds', fontsize=12, fontweight='bold', pad=15)
ax2.set_xlabel('Debate Rounds', fontsize=10)
ax2.set_ylabel('Avg Latency (seconds)', fontsize=10)
ax2.set_xticks([1, 2, 3])
ax2.set_ylim(0, 40)

# Add latency values
for x, y in zip(summary['rounds'], summary['Latency (s)']):
    ax2.text(x-1, y + 2, f'{y:.1f}s', ha='center', color='#cd5c5c', fontweight='bold')

# Global title
plt.suptitle('Layer 3 Optimization: Determining Optimal Debate Length', fontsize=14, y=1.05)

plt.tight_layout()
plt.savefig('results/debate_rounds_analysis.pdf', bbox_inches='tight')
print("Plot saved to results/debate_rounds_analysis.pdf")

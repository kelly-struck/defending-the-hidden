import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
INPUT_FILE = r'd:\AgentScope\results\efficiency_experiment.csv'
OUTPUT_FILE = r'd:\AgentScope\results\efficiency_analysis.pdf'

def setup_academic_style():
    """Sets up a publication-quality style for matplotlib."""
    # Manual fallback for academic style
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.facecolor'] = 'white'
    # plt.rcParams['axes.edgecolor'] = '#333333' # Keep default black
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    # Serif fonts for Latex feel
    plt.rcParams['font.family'] = 'serif' 
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 300

def plot_efficiency():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    setup_academic_style()
    df = pd.read_csv(INPUT_FILE)
    
    # Aggregation
    summary = df.groupby(['mode', 'type'])[['latency_ms', 'cost_rmb']].mean().reset_index()
    
    # Types and Modes
    type_order = ["Known Attack", "Replay Attack", "Unknown Sample"]
    modes = ["Pure Debate", "Cascade Defense"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Premium Colors: Deep Blue and Green
    colors = ["#1f4e79", "#2ca02c"] # Navy Blue (Baseline) vs Green (Method)
    
    x = np.arange(len(type_order))
    width = 0.35
    
    # --- Plot 1: Latency (Log Scale) ---
    ax1 = axes[0]
    for i, mode in enumerate(modes):
        subset = summary[summary['mode'] == mode]
        # Reindex to ensure order
        subset = subset.set_index('type').reindex(type_order).reset_index()
        vals = subset['latency_ms'].values
        
        # Add slight hatch pattern for distinction in B&W
        hatch = ''

        rects = ax1.bar(x + (i - 0.5) * width + width/2, vals, width,
                        label=mode, color=colors[i], edgecolor='black', alpha=0.9, hatch=hatch)
        
        # Value Labels
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                # Adapts label precision for low values (log scale)
                if height < 1:
                    label_text = f'{height:.1f}'
                else:
                    label_text = f'{height:.0f}'
                    
                ax1.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Latency (ms) [Log Scale]')
    ax1.set_title('Inference Latency Comparison')
    ax1.set_xticks(x + width / 2 if len(modes) > 1 else x)
    ax1.set_xticklabels(type_order)
    
    # Log scale handling with clean ticks
    ax1.set_yscale('log')
    # Use a lower bottom limit to show small values (~0.1ms)
    ax1.set_ylim(bottom=0.05) 
    ax1.yaxis.grid(False, which='major')
    ax1.yaxis.grid(False, which='minor')

    # --- Plot 2: Cost ---
    ax2 = axes[1]
    for i, mode in enumerate(modes):
        subset = summary[summary['mode'] == mode]
        subset = subset.set_index('type').reindex(type_order).reset_index()
        vals = subset['cost_rmb'].values
        
        hatch = ''

        rects = ax2.bar(x + (i - 0.5) * width + width/2, vals, width,
                        label=mode, color=colors[i], edgecolor='black', alpha=0.9, hatch=hatch)

        # Value Labels
        for rect in rects:
            height = rect.get_height()
            if not np.isnan(height):
                ax2.annotate(f'¥{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Cost (RMB)')
    ax2.set_title('Inference Cost Comparison')
    ax2.set_xticks(x + width / 2 if len(modes) > 1 else x)
    ax2.set_xticklabels(type_order)
    ax2.yaxis.grid(False)

    # Unified Legend
    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300)
    print(f"Generated clean academic plot: {OUTPUT_FILE}")

if __name__ == "__main__":
    plot_efficiency()

import json
import matplotlib.pyplot as plt
import numpy as np

# Configs
INPUT_FILE = r"d:\AgentScope\results\threshold_scores.json"
OUTPUT_FILE = r"d:\AgentScope\paper\graph\threshold_analysis.pdf"

def setup_academic_style():
    """Sets up a publication-quality style for matplotlib."""
    plt.rcParams['figure.figsize'] = (10, 5)
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

def main():
    setup_academic_style()

    # 1. Load Data
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    benign = np.array(data["benign_scores"])
    attack = np.array(data["attack_scores"])
    
    benign = np.clip(benign, 0, 1)
    attack = np.clip(attack, 0, 1)

    print(f"Loaded {len(benign)} Benign and {len(attack)} Attack scores.")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Plot 1: Histogram ---
    bins = np.linspace(0.0, 1.0, 21)
    weights_benign = np.ones_like(benign) / len(benign)
    weights_attack = np.ones_like(attack) / len(attack)

    # Colors: Navy Blue (Benign) vs Deep Red (Attack)
    color_benign = '#1f4e79' 
    color_attack = '#d62728' # Standard Red

    ax1.hist(benign, bins=bins, alpha=0.7, label='Benign (MME)', color=color_benign, edgecolor='black', linewidth=0.8, weights=weights_benign)
    ax1.hist(attack, bins=bins, alpha=0.6, label='Attack (SafetyBench)', color=color_attack, edgecolor='black', linewidth=0.8, weights=weights_attack)
    
    ax1.set_title('Distribution of Similarity Scores (Layer 2)', fontweight='bold')
    ax1.set_xlabel('Cosine Similarity Score')
    ax1.set_ylabel('Proportion')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Threshold Line
    ax1.axvline(0.85, color='black', linestyle='--', linewidth=1.5)
    # Optimized label placement with background box to avoid clutter/clipping
    ax1.text(0.85, ax1.get_ylim()[1]*0.95, 'Threshold: 0.85', 
             ha='center', va='top', color='black', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.9))

    # --- Plot 2: Recall vs FPR ---
    thresholds = np.linspace(0, 1, 101)
    tprs = [] # Recall (Sensitivity) = TP / (TP + FN) [Attack detected as Attack]
    fprs = [] # False Positive Rate = FP / (FP + TN) [Benign detected as Attack]
              # Note: Here "Positive" class is ATTACK
    
    for t in thresholds:
        tp = np.sum(attack >= t)
        fn = np.sum(attack < t)
        fp = np.sum(benign >= t)
        tn = np.sum(benign < t)
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tprs.append(tpr)
        fprs.append(fpr)
        
    # Plot curves
    ax2.plot(thresholds, tprs, label='Recall (Attack Detection)', color='#2ca02c', linewidth=2) # Green
    ax2.plot(thresholds, fprs, label='False Positive Rate (Benign blocked)', color='#d62728', linewidth=2) # Red
    
    ax2.set_title('Defense Performance vs. Threshold', fontweight='bold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Rate')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Highlight points
    points = [0.8, 0.85, 0.9]
    for p in points:
        idx = int(p * 100)
        if idx < len(thresholds):
            rec = tprs[idx]
            fp_rate = fprs[idx]
            
            # Draw marker
            ax2.plot(p, rec, 'o', color='black', markersize=5)
            
            # Label
            if p == 0.85:
                label = f"Optimal ({p})\nRecall: {rec:.2f}\nFPR: {fp_rate:.2f}"
                # Move label to Left-Up to avoid right boundary clipping
                ax2.annotate(label, xy=(p, rec), xytext=(-60, 40), textcoords='offset points', 
                             arrowprops=dict(arrowstyle='-|>', connectionstyle='arc3,rad=0.2', color='black', lw=1.5, mutation_scale=15), 
                             fontsize=10, fontweight='bold', ha='right')
            elif p == 0.8:
                 # Minimal label for context
                 pass 
            elif p == 0.9:
                 pass

    ax2.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    print(f"Generated academic plot: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

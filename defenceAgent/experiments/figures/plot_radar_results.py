import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Configure font for better aesthetics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']

def plot_radar_chart():
    # --- Data Preparation ---
    categories = [
        'Self-harm', 'Dangerous\nBehavior', 'Morality', 
        'Illegal\nActivities', 'Controversial\nTopics', 
        'Discrimination', 'Religion', 'Info\nMisinterpretation', 'Privacy'
    ]
    N = len(categories)

    # Totals for each category
    totals = np.array([19, 16, 20, 19, 17, 18, 20, 17, 21])

    # Method Data (Success Counts)
    data = {
        "Ours": np.array([18, 15, 18, 19, 15, 18, 20, 15, 19]),
        "Adashield": np.array([15, 11, 12, 14, 12, 14, 15, 13, 9]),
        "ECSO": np.array([6, 7, 4, 6, 9, 5, 3, 7, 7]),
        "Vanilla (Qwen-VL-Plus)": np.array([5, 4, 4, 5, 9, 5, 3, 7, 7]),
        "Vanilla (Doubao-1.5)": np.array([9, 4, 4, 5, 12, 10, 13, 10, 7])
    }

    # Colors and Styles
    style_config = {
        "Ours": {"color": "#E55039", "style": "-", "width": 3.0, "fill": True, "alpha": 0.2}, # Strong Red/Orange
        "Adashield": {"color": "#3C40C6", "style": "--", "width": 2.0, "fill": False, "alpha": 0.0}, # Deep Blue
        "ECSO": {"color": "#F39C12", "style": "--", "width": 2.0, "fill": False, "alpha": 0.0}, # Orange/Gold
        "Vanilla (Qwen-VL-Plus)": {"color": "#009432", "style": ":", "width": 2.5, "fill": False, "alpha": 0.0}, # Strong Green
        "Vanilla (Doubao-1.5)": {"color": "#5758BB", "style": ":", "width": 2.5, "fill": False, "alpha": 0.0} # Deep Purple
    }

    # Calculate Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop

    # Initialize Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Rotate plot so first axis is at top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, size=11, weight='bold')
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=9)
    plt.ylim(0, 1.05)

    # Plot Data
    for method_name, counts in data.items():
        # Calculate rates
        values = counts / totals
        values = np.concatenate((values, [values[0]])) # Close the loop
        
        config = style_config[method_name]
        
        ax.plot(angles, values, 
                linewidth=config["width"], 
                linestyle=config["style"], 
                label=method_name, 
                color=config["color"])
        
        if config["fill"]:
            ax.fill(angles, values, color=config["color"], alpha=config["alpha"])

    # Add Legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10, frameon=False)
    
    # Title
    plt.title("Defense Success Rate on SIUO Dataset", size=16, weight='bold', y=1.08)

    # Aesthetics
    ax.spines['polar'].set_visible(False) # Hide chart border
    ax.grid(color='#ECECEC', linestyle='-', linewidth=1.5) # Soft grid
    ax.set_facecolor('#FAFAFA') # Very light background for the circle

    # Save
    output_path = r"D:\AgentScope\els-cas-templates\graph\defense_radar_chart.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    plot_radar_chart()

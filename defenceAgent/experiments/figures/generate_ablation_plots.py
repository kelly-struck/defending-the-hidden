from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any


def generate_ablation_plots(results: Dict[str, Dict[str, Any]], output_dir: str = "d:/AgentScope/paper/graph"):
    """Generate publication-quality ablation plots.
    
    Args:
        results: Dict of experiment_name -> metrics dict
        output_dir: Directory to save PDF plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract core ablation data
    core_configs = [
        ("Full System", "full_system"),
        ("w/o-Attacker", "wo_attacker"),
        ("w/o-Defender", "wo_defender"),
        ("w/o-Asymmetric Rule", "wo_asymmetric"),
        ("w/o-Intent-as-Action", "wo_intent_as_action"),
        ("w/o-Retrieval", "wo_retrieval"),
    ]
    
    core_labels = []
    core_dsr = []
    core_asr = []
    core_tokens = []
    
    for label, key in core_configs:
        if key in results:
            core_labels.append(label)
            core_dsr.append(results[key]["dsr"] * 100)
            core_asr.append(results[key]["asr"] * 100)
            core_tokens.append(results[key]["avg_tokens"])
    
    # Plot 1: DSR comparison (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(core_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, core_dsr, width, label="DSR (Defense Success Rate)", color="#2E86AB")
    bars2 = ax.bar(x + width/2, core_asr, width, label="ASR (Attack Success Rate)", color="#A23B72")
    
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Ablation Study: Core Components", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(core_labels, rotation=15, ha="right", fontsize=10)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_core_dsr.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 2: Token cost comparison
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2E86AB" if "Full" in label else "#F18F01" for label in core_labels]
    bars = ax.barh(core_labels, core_tokens, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Average Tokens per Sample", fontsize=12)
    ax.set_title("Ablation Study: Computational Cost", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.0f}", xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_core_tokens.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 3: Round ablation (line plot)
    round_labels = []
    round_dsr = []
    round_asr = []
    round_tokens = []
    
    for rounds in [1, 2, 3, 4]:
        key = f"rounds_{rounds}"
        if key in results:
            round_labels.append(f"{rounds}R")
            round_dsr.append(results[key]["dsr"] * 100)
            round_asr.append(results[key]["asr"] * 100)
            round_tokens.append(results[key]["avg_tokens"])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # DSR vs rounds
    ax1.plot(round_labels, round_dsr, marker="o", linewidth=2, markersize=8, color="#2E86AB", label="DSR")
    ax1.plot(round_labels, round_asr, marker="s", linewidth=2, markersize=8, color="#A23B72", label="ASR")
    ax1.set_xlabel("Debate Rounds", fontsize=12)
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.set_title("Performance vs. Rounds", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Tokens vs rounds
    ax2.plot(round_labels, round_tokens, marker="D", linewidth=2, markersize=8, color="#F18F01")
    ax2.set_xlabel("Debate Rounds", fontsize=12)
    ax2.set_ylabel("Average Tokens", fontsize=12)
    ax2.set_title("Cost vs. Rounds", fontsize=13, fontweight="bold")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_rounds.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Plot 4: Temperature ablation
    temp_labels = []
    temp_dsr = []
    temp_asr = []
    
    for temp in [0.3, 0.5, 0.7]:
        key = f"temp_{temp}"
        if key in results:
            temp_labels.append(f"T={temp}")
            temp_dsr.append(results[key]["dsr"] * 100)
            temp_asr.append(results[key]["asr"] * 100)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(temp_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, temp_dsr, width, label="DSR", color="#2E86AB")
    bars2 = ax.bar(x + width/2, temp_asr, width, label="ASR", color="#A23B72")
    
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_title("Temperature Ablation", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(temp_labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "ablation_temperature.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Plots saved to {output_path}")
    return output_path


def generate_latex_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Generate LaTeX table for ablation results."""
    lines = [
        "% Ablation Study Results - Auto-generated",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation Study on SIUO Dataset (167 harmful samples). DSR=Defense Success Rate, ASR=Attack Success Rate.}",
        "\\label{tab:ablation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{DSR (\\%)} & \\textbf{ASR (\\%)} & \\textbf{Avg Tokens} & \\textbf{Avg Time (s)} \\\\",
        "\\midrule",
    ]
    
    # Core ablations
    core_configs = [
        ("Full System", "full_system"),
        ("w/o-Attacker", "wo_attacker"),
        ("w/o-Defender", "wo_defender"),
        ("w/o-Asymmetric Rule", "wo_asymmetric"),
        ("w/o-Intent-as-Action", "wo_intent_as_action"),
        ("w/o-Retrieval", "wo_retrieval"),
    ]
    
    for label, key in core_configs:
        if key in results:
            m = results[key]
            lines.append(
                f"{label} & {m['dsr']*100:.1f} & {m['asr']*100:.1f} & {m['avg_tokens']:.0f} & {m['avg_time_sec']:.2f} \\\\"
            )
    
    lines.append("\\midrule")
    
    # Round ablations
    for rounds in [1, 2, 3, 4]:
        key = f"rounds_{rounds}"
        if key in results:
            m = results[key]
            lines.append(
                f"{rounds}-Round Debate & {m['dsr']*100:.1f} & {m['asr']*100:.1f} & {m['avg_tokens']:.0f} & {m['avg_time_sec']:.2f} \\\\"
            )
    
    lines.append("\\midrule")
    
    # Temperature ablations
    for temp in [0.3, 0.5, 0.7]:
        key = f"temp_{temp}"
        if key in results:
            m = results[key]
            lines.append(
                f"Temperature {temp} & {m['dsr']*100:.1f} & {m['asr']*100:.1f} & {m['avg_tokens']:.0f} & {m['avg_time_sec']:.2f} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def save_ablation_artifacts(results: Dict[str, Dict[str, Any]], output_dir: str = "d:/AgentScope/results/ablation"):
    """Save all ablation artifacts: JSON, LaTeX table, and plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    import json
    with open(output_path / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save LaTeX table
    latex = generate_latex_table(results)
    with open(output_path / "ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    
    # Generate plots
    generate_ablation_plots(results, output_dir="d:/AgentScope/paper/graph")
    
    print(f"All artifacts saved. Results: {output_path}, Plots: d:/AgentScope/paper/graph")
    return output_path


if __name__ == "__main__":
    # Example: generate plots from existing results
    import json
    results_path = Path("d:/AgentScope/results/ablation/ablation_results.json")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        generate_ablation_plots(results)
        latex = generate_latex_table(results)
        print(latex)
    else:
        print("No results file found. Run run_ablation_study.py first.")

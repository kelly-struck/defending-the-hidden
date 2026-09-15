import json
import os
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Paths
FPR_DIR = Path("d:/AgentScope/results/fpr_analysis")
HUMAN_VAL_DIR = Path("d:/AgentScope/results/human_validation")
OUTPUT_DIR = Path("d:/AgentScope/results/fpr_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR = Path("d:/AgentScope/paper/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def load_fpr_results() -> tuple[List[Dict], Dict]:
    """Load FPR evaluation results."""
    raw_path = FPR_DIR / "fpr_raw_results.json"
    stats_path = FPR_DIR / "fpr_statistics.json"
    
    if not raw_path.exists() or not stats_path.exists():
        print(f"Warning: FPR results not found in {FPR_DIR}")
        return [], {}
    
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    return raw, stats


def load_human_validation_results() -> tuple[List[Dict], Dict]:
    """Load human validation results."""
    raw_path = HUMAN_VAL_DIR / "human_validation_results.json"
    stats_path = HUMAN_VAL_DIR / "validation_statistics.json"
    
    if not raw_path.exists() or not stats_path.exists():
        print(f"Warning: Human validation results not found in {HUMAN_VAL_DIR}")
        return [], {}
    
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    return raw, stats


def analyze_false_positives(raw_results: List[Dict], stats: Dict) -> Dict[str, Any]:
    """Analyze false positive patterns and causes."""
    fp_analysis = {
        "total_fp": len(stats.get("false_positives", [])),
        "by_domain": {},
        "by_modality": {},
        "by_source": defaultdict(int),
        "semantic_overlap_cases": [],
        "domain_ambiguity_cases": [],
        "modality_complexity_cases": [],
    }
    
    # Get all FP samples
    fp_samples = [r for r in raw_results if not r.get("predicted_safe", True)]
    
    # Categorize FPs
    for sample in fp_samples:
        domain = sample.get("domain", "unknown")
        modality = sample.get("modality", "unknown")
        source = sample.get("source", "unknown")
        question = sample.get("question", "")
        
        fp_analysis["by_source"][source] += 1
        
        # Detect semantic overlap (benign query that resembles harmful)
        overlap_keywords = [
            "kill", "die", "death", "harm", "dangerous", "weapon", "poison",
            "hack", "steal", "illegal", "drug", "bomb", "attack", "violence",
            "suicide", "self-harm", "murder", "crime", "terrorist"
        ]
        if any(kw in question.lower() for kw in overlap_keywords):
            fp_analysis["semantic_overlap_cases"].append({
                "id": sample.get("sample_id"),
                "question": question,
                "reason": "Query contains terms that semantically overlap with harmful concepts",
            })
        
        # Detect domain ambiguity (queries in gray-area domains)
        ambiguous_domains = ["technical", "creative"]
        if domain in ambiguous_domains:
            fp_analysis["domain_ambiguity_cases"].append({
                "id": sample.get("sample_id"),
                "domain": domain,
                "question": question,
                "reason": f"Domain '{domain}' has inherent ambiguity between benign and harmful usage",
            })
        
        # Detect modality complexity (multi-modal inputs)
        if modality in ["multi_image_text"]:
            fp_analysis["modality_complexity_cases"].append({
                "id": sample.get("sample_id"),
                "modality": modality,
                "question": question,
                "reason": "Multi-modal input increases complexity and may trigger false alarms",
            })
    
    fp_analysis["by_source"] = dict(fp_analysis["by_source"])
    
    return fp_analysis


def generate_latex_tables(stats: Dict, fp_analysis: Dict) -> str:
    """Generate LaTeX table code for paper."""
    latex_code = []
    
    # Table 1: Overall FPR
    latex_code.append("% Table: Overall FPR Results")
    latex_code.append("\\begin{table}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{False Positive Rate (FPR) on Benign Dataset}")
    latex_code.append("\\begin{tabular}{lcc}")
    latex_code.append("\\toprule")
    latex_code.append("Metric & Value & Count \\\\")
    latex_code.append("\\midrule")
    
    total = stats.get("total_samples", 0)
    fpr = stats.get("overall_fpr", 0)
    fp_count = int(fpr * total)
    latex_code.append(f"Overall FPR & {fpr:.4f} & {fp_count}/{total} \\\\")
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\label{tab:fpr_overall}")
    latex_code.append("\\end{table}")
    latex_code.append("")
    
    # Table 2: FPR by Domain
    latex_code.append("% Table: FPR by Domain")
    latex_code.append("\\begin{table}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{FPR by Domain}")
    latex_code.append("\\begin{tabular}{lccc}")
    latex_code.append("\\toprule")
    latex_code.append("Domain & Total & False Positives & FPR \\\\")
    latex_code.append("\\midrule")
    
    for domain, dstats in sorted(stats.get("by_domain", {}).items()):
        latex_code.append(
            f"{domain} & {dstats['total']} & {dstats['false_positives']} & {dstats['fpr']:.4f} \\\\"
        )
    
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\label{tab:fpr_by_domain}")
    latex_code.append("\\end{table}")
    latex_code.append("")
    
    # Table 3: FPR by Modality
    latex_code.append("% Table: FPR by Modality")
    latex_code.append("\\begin{table}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{FPR by Modality Complexity}")
    latex_code.append("\\begin{tabular}{lccc}")
    latex_code.append("\\toprule")
    latex_code.append("Modality & Total & False Positives & FPR \\\\")
    latex_code.append("\\midrule")
    
    for mod, mstats in sorted(stats.get("by_modality", {}).items()):
        latex_code.append(
            f"{mod} & {mstats['total']} & {mstats['false_positives']} & {mstats['fpr']:.4f} \\\\"
        )
    
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\label{tab:fpr_by_modality}")
    latex_code.append("\\end{table}")
    latex_code.append("")
    
    # Table 4: Human Validation
    latex_code.append("% Table: Human Validation Results")
    latex_code.append("\\begin{table}[t]")
    latex_code.append("\\centering")
    latex_code.append("\\caption{Human Validation Results (300 Samples)}")
    latex_code.append("\\begin{tabular}{lcc}")
    latex_code.append("\\toprule")
    latex_code.append("Metric & Value \\\\")
    latex_code.append("\\midrule")
    
    # This will be filled when human validation stats are available
    latex_code.append("Overall Agreement & TBD \\\\")
    latex_code.append("Cohen's Kappa ($\\kappa$) & TBD \\\\")
    latex_code.append("True Positives & TBD \\\\")
    latex_code.append("False Positives & TBD \\\\")
    latex_code.append("True Negatives & TBD \\\\")
    latex_code.append("False Negatives & TBD \\\\")
    
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\label{tab:human_validation}")
    latex_code.append("\\end{table}")
    
    return "\n".join(latex_code)


def generate_plots(stats: Dict, output_dir: Path):
    """Generate plots for FPR analysis."""
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    
    # Plot 1: FPR by Domain (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    domains = []
    fprs = []
    for domain, dstats in sorted(stats.get("by_domain", {}).items()):
        domains.append(domain)
        fprs.append(dstats['fpr'])
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(domains, fprs, color=colors[:len(domains)], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_xlabel('Domain', fontsize=12)
    ax.set_title('FPR by Domain', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(fprs + [0.1]) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_by_domain.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fpr_by_domain.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: FPR by Modality (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    modalities = []
    fprs = []
    for mod, mstats in sorted(stats.get("by_modality", {}).items()):
        modalities.append(mod.replace('_', ' ').title())
        fprs.append(mstats['fpr'])
    
    colors = ['#9b59b6', '#1abc9c', '#e67e22']
    bars = ax.bar(modalities, fprs, color=colors[:len(modalities)], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_xlabel('Modality Complexity', fontsize=12)
    ax.set_title('FPR by Modality Complexity', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(fprs + [0.1]) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fpr_by_modality.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fpr_by_modality.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Domain-Modality Heatmap
    domain_modality = stats.get("by_domain_modality", {})
    if domain_modality:
        domains = sorted(set(d for d in domain_modality.keys()))
        modalities = sorted(set(m for mods in domain_modality.values() for m in mods.keys()))
        
        matrix = np.zeros((len(domains), len(modalities)))
        for i, domain in enumerate(domains):
            for j, modality in enumerate(modalities):
                matrix[i, j] = domain_modality.get(domain, {}).get(modality, 0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(np.arange(len(modalities)))
        ax.set_yticks(np.arange(len(domains)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in modalities], rotation=45, ha='right')
        ax.set_yticklabels([d.title() for d in domains])
        
        ax.set_xlabel('Modality Complexity', fontsize=12)
        ax.set_ylabel('Domain', fontsize=12)
        ax.set_title('False Positive Distribution by Domain and Modality', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(domains)):
            for j in range(len(modalities)):
                text = ax.text(j, i, int(matrix[i, j]),
                             ha="center", va="center", color="black" if matrix[i, j] < matrix.max()/2 else "white",
                             fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('False Positive Count', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / "fpr_heatmap.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "fpr_heatmap.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def generate_human_validation_plots(val_stats: Dict, output_dir: Path):
    """Generate plots for human validation results."""
    if not val_stats:
        return
    
    # Plot 1: Agreement by Category
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = []
    agreements = []
    
    for cat, cstats in sorted(val_stats.get("by_category", {}).items()):
        categories.append(cat)
        agreements.append(cstats['agreement'])
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.barh(categories, agreements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Agreement Rate', fontsize=12)
    ax.set_ylabel('SIUO Category', fontsize=12)
    ax.set_title('System-Human Agreement by SIUO Category', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='High Agreement Threshold (0.8)')
    ax.legend()
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "human_agreement_by_category.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "human_agreement_by_category.png", format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Judgment Distribution (Pie Chart)
    fig, ax = plt.subplots(figsize=(7, 7))
    judgments = val_stats.get("judgment_distribution", {})
    if judgments:
        labels = list(judgments.keys())
        sizes = list(judgments.values())
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
                                           autopct='%1.1f%%', startangle=90,
                                           textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Distribution of Judgment Types (300 Samples)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "judgment_distribution.pdf", format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / "judgment_distribution.png", format='png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Human validation plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FPR and human validation results")
    parser.add_argument("--fpr-dir", type=str, default=str(FPR_DIR))
    parser.add_argument("--human-val-dir", type=str, default=str(HUMAN_VAL_DIR))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--graph-dir", type=str, default=str(GRAPH_DIR))
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    graph_dir = Path(args.graph_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FPR and Human Validation Analysis")
    print("=" * 60)
    
    # Load FPR results
    print("\n[1/4] Loading FPR results...")
    fpr_raw, fpr_stats = load_fpr_results()
    
    if fpr_stats:
        print(f"      Loaded {fpr_stats.get('total_samples', 0)} FPR evaluation results")
    
    # Analyze false positives
    print("[2/4] Analyzing false positive patterns...")
    fp_analysis = analyze_false_positives(fpr_raw, fpr_stats)
    
    # Load human validation results
    print("[3/4] Loading human validation results...")
    val_raw, val_stats = load_human_validation_results()
    
    if val_stats:
        print(f"      Loaded {val_stats.get('total_samples', 0)} validation results")
    
    # Generate LaTeX tables
    print("[4/4] Generating LaTeX tables and plots...")
    latex_code = generate_latex_tables(fpr_stats, fp_analysis)
    
    with open(output_dir / "latex_tables.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"      LaTeX tables saved to {output_dir / 'latex_tables.tex'}")
    
    # Generate plots
    if fpr_stats:
        generate_plots(fpr_stats, graph_dir)
    
    if val_stats:
        generate_human_validation_plots(val_stats, graph_dir)
    
    # Save comprehensive analysis report
    with open(output_dir / "comprehensive_analysis.json", "w", encoding="utf-8") as f:
        analysis = {
            "fpr_statistics": fpr_stats,
            "false_positive_analysis": fp_analysis,
            "human_validation_statistics": val_stats,
            "summary": {
                "overall_fpr": fpr_stats.get("overall_fpr", 0),
                "human_agreement": val_stats.get("overall_agreement", 0),
                "cohens_kappa": val_stats.get("cohens_kappa", 0),
            }
        }
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("Analysis Summary")
    print(f"{'=' * 60}")
    if fpr_stats:
        print(f"Overall FPR: {fpr_stats.get('overall_fpr', 0):.4f}")
    if val_stats:
        print(f"Human Agreement: {val_stats.get('overall_agreement', 0):.4f}")
        print(f"Cohen's Kappa: {val_stats.get('cohens_kappa', 0):.4f}")
    
    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

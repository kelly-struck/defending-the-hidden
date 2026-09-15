from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_siuo_categories(siuo_json: Path) -> dict[int, str]:
    """Map 0-based SIUO row index -> category.

    NOTE:
    In `results/debate_rounds_experiment.csv`, `sample_id` is the row index used
    during the experiment loop, not SIUO's `question_id` (1001, 1002, ...).
    """
    data = json.loads(siuo_json.read_text(encoding="utf-8"))
    return {idx: item.get("category", "unknown") for idx, item in enumerate(data)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize SIUO debate rounds (R1/R2/R3) by category"
    )
    parser.add_argument(
        "--rounds_csv",
        type=str,
        default=r"d:\AgentScope\results\debate_rounds_experiment.csv",
        help="Path to debate_rounds_experiment.csv",
    )
    parser.add_argument(
        "--siuo_json",
        type=str,
        default=r"d:\AgentScope\defenceAgent\statics\SIUO\siuo_gen.json",
        help="Path to SIUO generation json",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=r"d:\AgentScope\results\debate_rounds_by_category.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out_md",
        type=str,
        default=r"d:\AgentScope\results\debate_rounds_by_category.md",
        help="Output Markdown table path",
    )
    args = parser.parse_args()

    rounds_csv = Path(args.rounds_csv)
    siuo_json = Path(args.siuo_json)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    if not rounds_csv.is_file():
        raise FileNotFoundError(f"rounds csv not found: {rounds_csv}")
    if not siuo_json.is_file():
        raise FileNotFoundError(f"siuo json not found: {siuo_json}")

    df = pd.read_csv(rounds_csv)
    required_cols = {"rounds", "is_blocked", "sample_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    category_map = load_siuo_categories(siuo_json)
    df["category"] = df["sample_id"].map(category_map)

    # Interception rate by category/round
    table = (
        df.groupby(["category", "rounds"], dropna=False)["is_blocked"]
        .mean()
        .unstack(fill_value=float("nan"))
        * 100.0
    )

    # Make sure we always expose 1/2/3 columns in order.
    for r in [1, 2, 3]:
        if r not in table.columns:
            table[r] = float("nan")
    table = table[[1, 2, 3]]
    table.columns = ["Round 1 (%)", "Round 2 (%)", "Round 3 (%)"]

    # Add sample counts by category (records per category across all rounds)
    n_total = df.groupby("category", dropna=False).size().rename("N records")
    table = table.join(n_total, how="left")

    # Add overall row
    overall = (df.groupby("rounds")["is_blocked"].mean() * 100.0).to_dict()
    table.loc["Overall", "Round 1 (%)"] = overall.get(1)
    table.loc["Overall", "Round 2 (%)"] = overall.get(2)
    table.loc["Overall", "Round 3 (%)"] = overall.get(3)
    table.loc["Overall", "N records"] = len(df)

    # Persist artifacts
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    table_rounded = table.copy()
    for c in ["Round 1 (%)", "Round 2 (%)", "Round 3 (%)"]:
        table_rounded[c] = table_rounded[c].round(2)

    table_rounded.to_csv(out_csv, encoding="utf-8")
    out_md.write_text(table_rounded.to_markdown(), encoding="utf-8")

    print("Saved:")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    print("\nPreview:")
    print(table_rounded.to_markdown())


if __name__ == "__main__":
    main()

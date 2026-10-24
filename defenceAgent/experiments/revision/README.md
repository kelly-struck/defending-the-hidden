# Revision-round experiments

Scripts added for the revision of *Assigning Suspicion* (NEUCOM-D-26-20854). Each
folder-level experiment below answers one referee point with a measurement, and
each entry in this file maps a script to the place in the manuscript where its
result is reported.

Only **code** is released here. Per-sample logs, result JSON/Markdown, generated
attack texts, dataset images/caches, the ChromaDB directories and the cached
embedding matrices stay out of the repository, exactly as for the rest of
`defenceAgent/experiments/`. Every script writes into `revision/results/`, which
is git-ignored.

| Experiment | Script(s) | What it measures | Reported in |
|---|---|---|---|
| **E1a** — adaptive attack on the published prompts | `e1a_adaptive_attack.py`, `e1a_prepare_data.py`, `e1a_check_fidelity.py`, `e1a_verify.py` | 50 HarmBench-standard intents × 2 vectors (benign visual distractor / scientific framing) × ≤4 revisions against a local replica of the full cascade; per-round L1/L2/L3 record, end-to-end victim answer judged by an independent oracle | §5.6, Table *Adaptive evasion* |
| **E1b** — evasion of consolidated memory | `e1b_threshold_evasion.py` | Three-stage protocol: consolidate blocked E1a attacks into an isolated collection → semantics-preserving rewrite until L2 no longer fires (image replacement as fallback) → replay the bypasses through the full cascade | §5.6, Table *Threshold evasion across the consolidation boundary* |
| **E1c** — complete benchmark report | `e1c_benchmark_full_report.py`, `e1c_verify.py`, `e1c_render_table.py` | Layer-3-only HarmBench (stratified *N* = 200) and MM-SafetyBench (*N* = 494) with Wilson intervals and per-class breakdown; L1/L2 disabled to avoid retrieval leakage | §5.2.1, Table 1 |
| **E2a** — disjoint threshold recalibration | `e2a_tau_recalibration.py`, `e2a_verify.py` | τ sweep on FigStep (strictly outside the seed library) against the 7,120-entry production library, plus leave-one-category-out folds | §5.4.3, Table *Threshold decision framework* |
| **E2b** — threshold decision framework | `e2b_tau_framework.py`, `e2b_audit.py` | Assembles the four operating points (production 0.85, matched-FPR sweep, retention 0.427, recalibrated τ\*) with their calibration distributions and objectives; adds the τ = 0.85 retention arm | §5.4.3, §5.5.2 |
| **E3** — retention-constant sensitivity and drift | `e3_run_ttl_simulation.py`, `e3_sensitivity_ext.py`, `e3_sensitivity_regression.py` | 56-cell grid over δ, quarantine threshold o<sub>e</sub>, pseudocounts α = β, five T<sub>min</sub>/T<sub>max</sub> spans, traffic mixes, rare-activation schedules and novel-traffic shares; fully offline over cached embeddings | Appendix D |
| **E4** — statistical margin and review cost | `e4_dsr_gap_ci.py`, `e4_review_cost.py`, `e4_recheck.py` | Newcombe exact CI for the 154/167 vs 148/167 difference; alerts per 1,000 queries and analyst hours at r ∈ {0.90, 0.95, 0.99, 0.999} under 3 min per alert | §5.8, §5.3.2 |
| **E5** — false-positive-induction probe | `e5_fp_induction_probe.py`, `e5_audit.py` | 50 celebrity-identification benign queries rewritten into 150 semantics-preserving variants against 50 matched controls, run through the real cascade | §5.3.2 |

## Running

All drivers resolve the repository root from their own location, so they are
launched from the repository root like the other experiment folders:

```bash
cd defending-the-hidden
python defenceAgent/experiments/revision/e1a_adaptive_attack.py --pilot --dry-run
python defenceAgent/experiments/revision/e1c_benchmark_full_report.py --help
python defenceAgent/experiments/revision/e2a_tau_recalibration.py --help
```

`verify`/`audit` companions recompute the quantities in a report from its
checkpoint without importing the executor, and exit non-zero on any mismatch;
they are the scripts used to check the numbers in the manuscript.

## Inputs that are not in this repository

These drivers were written against the authors' workspace; the *import* path is
fixed but the *data* path is not. Point the constants (or the flags) at your own
copies before an end-to-end run:

| Script | Needs |
|---|---|
| `e1a_adaptive_attack.py`, `e1a_prepare_data.py` | HarmBench `standard` parquet, MME images (benign distractors), `chroma_db_phase1` seed library, `DASHSCOPE_API_KEY`; an oracle key for the end-to-end arm (`DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY` or the qwen-plus path) |
| `e1b_threshold_evasion.py` | An E1a checkpoint (default `revision/results/e1a_ckpt.jsonl`); writes to its own `chroma_db_e1b` persist directory and `e1b_adaptive` collection |
| `e1c_benchmark_full_report.py` | HarmBench parquet and MM-SafetyBench categories under `defenceAgent/statics/`, `DASHSCOPE_API_KEY` |
| `e2a_tau_recalibration.py`, `e2b_tau_framework.py` | `Adashield/data/datasplit/figstep.jsonl`, `datasets/benign/benign_dataset.json`, `chroma_db_phase1` |
| `e3_*` | `ESWA_submission/ttl_experiment/embeddings/*.npy` (built by `e3_run_ttl_simulation.py --phase embed`), SIUO case list, MM-SafetyBench data, benign set |
| `e5_fp_induction_probe.py` | Benign set, `chroma_db` production collection, `DASHSCOPE_API_KEY` |

## Isolation and disclosure

* E1a and E1b never write to the production library: E1a uses a dedicated L1
  cache file, E1b a separate persist directory and collection, E5 a dedicated L1
  cache.
* E1a regenerates its attack texts at run time; the released paper reports
  aggregate rates and desensitised examples only. The harness is published
  because §3 of the paper adopts a white-box adversary — the prompts are assumed
  known — and the measurement is part of the contribution. It is not a
  general-purpose jailbreak generator, and no generated attack corpus is
  redistributed.
* E1c runs the debate gate with `learning_enabled = False`, so evaluation cannot
  write back into the seed library.

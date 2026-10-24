# Experiment code

Every run behind the paper's Experiments section. All scripts were written in the
authors' research workspace and are published here for inspection and reuse; the
grouping follows the paper's structure so a reader can move from a table or
figure to the code that produced it.

```
experiments/
├── data_prep/            datasets and the seed threat library
├── siuo/                 the primary SIUO testbed
├── baselines/            the five inference-time baselines
├── standard_benchmarks/  HarmBench / MM-SafetyBench / FigStep harness
├── utility/              MME and MM-Vet capability preservation
├── ablation/             component ablation, debate rounds, τ calibration
├── fpr/                  false-positive study and the matched-FPR ROC
├── consolidation/        verdict consolidation across threat waves
├── poisoning/            library-corruption simulation
├── efficiency/           latency and token accounting
├── statistics/           significance tests, κ, recomputation
├── figures/              figure generators
└── revision/             experiments added for the revision round (§5.6, §5.4.3, §5.3.2, App. D)
```

## Layout conventions

Scripts here live exactly **two levels below `defenceAgent/`**, and import the
package from the repository root:

```python
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..")))  # repo root
```

Run them from the repository root, e.g.

```bash
cd defending-the-hidden
python defenceAgent/experiments/siuo/run_siuo_with_records.py --help
```

> ⚠️ **Data paths.** These drivers were written against the authors' local layout
> and several carry absolute paths to datasets, result files and the ChromaDB
> directory. The import path is fixed, but you will typically need to point a
> script's input/output constants (or `--output` / `--output-dir` flag) at your
> own copy before it runs end to end. The `revision/` drivers resolve the
> repository root from their own location and write into `revision/results/`, but
> they still expect the datasets, the production ChromaDB directory and the cached
> embeddings listed in [`revision/README.md`](revision/README.md). The reference
> pipeline itself (`defenceAgent/main.py`, `main_debate.py`,
> `defenceAgent/eval/code/*`) needs no such edits.

## What each script produced

### `data_prep/` — datasets and seed library (§ Experimental Setup, § Layer 2 seed library)

| Script | Role |
|---|---|
| `ingest_harmbench.py` | Ingests HarmBench (400 templates) into the Layer 2 vector library |
| `ingest_mmsafetybench.py` | Ingests MM-SafetyBench (6,720 scenarios) into the Layer 2 vector library |
| `getdata.py` | Downloads the MME dataset into `defenceAgent/statics/MME/` |
| `getdata1.py` | Downloads the SIUO image set into `defenceAgent/statics/SIUO/` |
| `build_benign_dataset.py` | Assembles the 1,000-sample benign set (MME 822 + MM-Vet 178) behind the FPR study |

HarmBench + MM-SafetyBench form the 7,120-entry seed library. **SIUO is
deliberately not ingested** — the 167 cases are the held-out probe.

### `siuo/` — implicit cross-modal threats (§ Implicit Threat Detection on SIUO)

| Script | Role |
|---|---|
| `run_siuo_with_records.py` | The paper's per-sample SIUO evaluator. One schema for all four arms (`full_system`, `single_agent_control`, `symmetric`, `no_defense`); writes per-sample JSONL with checkpoint/resume so a rerun does not re-spend API budget. Feeds the significance tests, the McNemar comparisons, the confusion matrix and the τ sweep |
| `run_no_defense.py` | The undefended arm: the victim model answers SIUO directly, judged by the independent oracle (44.7% on Doubao, 29.8% on Qwen-VL-Plus) |

The benchmark entry points live in `defenceAgent/eval/code/`:
`eval_siuo.py`, `eval_siuo_doubao_with_defense.py`,
`eval_siuo_doubao_no_defense.py`, `eval_siuo_judge_defender.py`,
`eval_siuo_ablation.py`, `summary_siuo.py`, `verify_doubao_vision.py`.

### `baselines/` — the five baselines (§ Victim Models and Baselines)

| Script | Role |
|---|---|
| `run_baseline_safety_prompt.py` | Enhanced Safety Prompting — the upper bound of single-prompt defense |
| `run_baseline_llm_guard.py` | LLM Guard — input/output screening, adapted from Llama Guard / MLLM-Protector |
| `run_baseline_cider.py` | CIDER (adapted) — cross-modal consistency checking |
| `reproduce_adashield.py` | AdaShield (static prompt) |
| `reproduce_adaptive_adashield.py` | AdaShield (dynamic / adaptive prompt) |
| `reproduce_ecso_siuo.py` | ECSO — captions the visual input to bypass visual jailbreaks |
| `evaluate_adashield_siuo_full.py` | Full-set AdaShield SIUO run |
| `evaluate_baselines.py` | Scores every baseline's output with the shared LLM safety classifier so DSR/ASR are computed on one definition |

All five are the authors' re-implementations under a common harness, not the
original authors' code.

### `standard_benchmarks/` — explicit and composite attacks (§ Safety on Standard Benchmarks)

Measured with Layer 3 debate alone, L1/L2 disabled, to avoid retrieval leakage.

| Script | Role |
|---|---|
| `run_experiment.py` | Driver for the HarmBench / MM-SafetyBench × victim model × baseline matrix |
| `run_all_experiments2.ps1` | Batch runner for that matrix |
| `adashield_utils.py` | AdaShield static/dynamic prompt variants |
| `eval_utils.py` | Dataset loading and typographic (FigStep-style) image rendering |
| `gpt_eval.py` | OpenAI-compatible ASR judging |

FigStep, HarmBench and MM-SafetyBench entry points are
`defenceAgent/eval/code/eval_figstep.py`, `eval_harmbench.py`,
`eval_mmsafetybench.py`, `eval_multimodal_suite.py`, `eval_compare.py`.

### `utility/` — capability preservation (§ Capability Preservation)

| Script | Role |
|---|---|
| `evaluate_utility_mme.py` | MME accuracy with and without the defense (net Yes/No accuracy) |
| `evaluate_utility_mmvet.py` | MM-Vet responses with and without the defense |
| `analyze_mme_results.py` | Aggregates the MME runs |

MM-Vet scoring helpers are `eval_mmvet.py`, `mmvet_official_api.py`,
`run_official_mmvet.py`, `eval_mmvet_official_only.py`, `grade_mmvet_local.py`
and `compare_mmvet_reports.py` in `defenceAgent/eval/code/`.

### `ablation/` — what carries the gain (§ Component Ablation, § Debate Round Analysis, § Retrieval Threshold Calibration)

| Script | Role |
|---|---|
| `run_ablation_enhanced.py` | The six core configurations: full system, w/o Attacker, w/o Defender, w/o Intent-as-Action, w/o Retrieval (L2), w/o Asymmetric Rule |
| `run_ablation_stages89.py` | Debate rounds (R1–R4) and temperature |
| `scan_debate_rounds.py` | Per-round DSR / token / latency scan |
| `scan_l2_threshold.py` | Layer 2 τ sweep on benign and attack traffic |
| `analyze_debate_rounds_by_category.py` | Per-category breakdown of the round configurations |

The ablation orchestrators themselves are `defenceAgent/orchestrator_ablation.py`.

### `fpr/` — false positives and the matched-FPR ROC (§ False Positive Rate Analysis)

| Script | Role |
|---|---|
| `evaluate_fpr_real.py` | Runs the real cascade over the 1,000 benign samples; the source of the by-domain and by-layer FPR tables |
| `run_single_agent_benign_fpr.py` | Benign-side FPR of the single-agent control on the matched subset |
| `run_matched_fpr_curve.py` | τ sweep producing the matched-FPR operating points for both arms |
| `analyze_fpr_results.py` | Aggregates runs into the domain and layer breakdowns |

### `consolidation/` — cost amortization (§ Multi-Round Iterative Validation)

| Script | Role |
|---|---|
| `evaluate_multi_round_evolution.py` | The three-wave experiment (physical → social → normative harm), raw and paraphrase phases |
| `evaluate_adaptive_evolution.py` | Repeated-wave consolidation variant |

### `poisoning/` — library-corruption simulation (§ Robustness against Knowledge Poisoning)

| Script | Role |
|---|---|
| `evaluate_poisoning_robustness.py` | 500-entry held-out library, 50 injected benign near-neighbours, comparing no-TTL / refresh-on-activation TTL / competence-based retention |
| `collect_real_l2_fps.py` | Collects the real Layer 2 celebrity-identification false positives used as poison seeds |

### `efficiency/` — latency and token accounting (§ Efficiency)

| Script | Role |
|---|---|
| `evaluate_efficiency.py` | Per-layer latency and token cost across replay / known-variant / novel traffic |

### `statistics/` — tests and recomputation

| Script | Role |
|---|---|
| `compute_significance_c1.py` | Three-seed DSR dispersion, per-domain Wilson intervals, bootstrap CIs, McNemar vs. the strongest baseline |
| `compute_confusion_a5.py` | Symmetric-rule vs. undefended confusion matrix |
| `compute_oracle_human_kappa.py` | Per-annotator oracle–human Cohen's κ alongside Fleiss' κ |
| `run_human_validation.py` | The three-annotator human-validation study |
| `reoracle_phase1.py` | Re-judges the `allow` responses with the qwen-plus cross-check oracle |
| `P0_recompute.py` | Independent recomputation of the load-bearing numbers (PPV/NPV, Wilson intervals, token arithmetic) |

### `figures/` — figure generators

| Script | Figure |
|---|---|
| `plot_radar_results.py` | Per-category DSR radar |
| `plot_debate_rounds.py` | Debate-round curve |
| `generate_ablation_plots.py` | Ablation bars |
| `make_fig_ablation_ladder.py` | Same-oracle ablation ladder |
| `plot_threshold_curve_v2.py` | Layer 2 threshold analysis |
| `plot_matched_fpr_curve.py` | Matched-FPR curve from the τ sweep |
| `plot_roc_matched_fpr_crossoracle.py` | Matched-FPR ROC with the cross-oracle control arm |
| `plot_efficiency_results_v2.py` | Latency and cost |
| `plot_multi_round_evolution.py` | Three-wave consolidation |
| `plot_poisoning_robustness.py` | Retention-policy comparison |

### `revision/` — measurements added for the revision (§5.6, §5.4.3, §5.3.2, Appendix D)

These answer the referees' requests for an adaptive evaluation, a disjoint
threshold recalibration, a retention-constant sweep, and a review-cost figure.
They follow the same layout convention as the folders above; see
[`revision/README.md`](revision/README.md) for the per-script input requirements.

| Script | Role |
|---|---|
| `revision/e1a_adaptive_attack.py` | Adaptive attack on the published Defender prompt and adjudication rule: 50 HarmBench-standard intents × 2 vectors × ≤4 revisions against a local replica of the cascade, with an end-to-end victim answer and oracle judgement |
| `revision/e1a_prepare_data.py`, `e1a_check_fidelity.py` | Builds the intent/image assignment; scores residual harmful intent in each generated vector |
| `revision/e1a_verify.py` | Recomputes the evasion rate, Wilson interval, per-layer split and harm rate from the raw checkpoint |
| `revision/e1b_threshold_evasion.py` | Three-stage evasion of *consolidated* memory: consolidate → rewrite below τ (image replacement fallback) → replay through L3 |
| `revision/e1c_benchmark_full_report.py` | Layer-3-only HarmBench (N=200) and MM-SafetyBench (N=494) report with Wilson intervals and per-class breakdown |
| `revision/e1c_verify.py`, `e1c_render_table.py` | Recomputes ASR/CI from the checkpoint; renders the LaTeX table |
| `revision/e2a_tau_recalibration.py` | τ recalibration on FigStep (out-of-library) plus leave-one-category-out folds |
| `revision/e2a_verify.py` | Recomputes the recalibration curves and τ\* from the embedding checkpoint |
| `revision/e2b_tau_framework.py` | Assembles the four-row threshold decision framework and the τ=0.85 retention arm |
| `revision/e2b_audit.py` | Independent audit of the framework table against its three source reports |
| `revision/e3_run_ttl_simulation.py` | Retention-policy simulator over the real embedding space (no-TTL / current-TTL / competence-TTL) |
| `revision/e3_sensitivity_ext.py` | 56-cell constant/drift grid over δ, quarantine threshold, pseudocounts, T<sub>min</sub>/T<sub>max</sub>, traffic mix, rare-activation schedule and novel share |
| `revision/e3_sensitivity_regression.py` | Regression check that the extended grid reproduces the original report on overlapping cells |
| `revision/e4_dsr_gap_ci.py` | Newcombe exact interval for the 154/167 vs 148/167 margin |
| `revision/e4_review_cost.py` | Alerts per 1,000 queries and analyst hours across deployment prevalence |
| `revision/e4_recheck.py` | Independent re-derivation of both, plus the admissible McNemar splits |
| `revision/e5_fp_induction_probe.py` | Directional false-positive-induction probe on the celebrity-identification hotspot, with 50 matched controls |
| `revision/e5_audit.py` | Recomputes arm counts, Wilson intervals and the two-proportion test from the raw checkpoint |

## Not included

Kept out on purpose: result files and per-sample outputs, dataset images and
parquet caches, the ChromaDB directory, runtime memory (`memory/*.jsonl`),
LaTeX build and manuscript-editing tooling, and superseded prototypes for which
a final version is present in this tree — `evaluate_adaptive_evolution_v1.py`,
`plot_efficiency_results.py`, `plot_threshold_curve.py`,
`plot_adaptive_evolution_only.py`, `reproduce_chart_v2.py`,
`evaluate_adashield_siuo.py`, `evaluate_fpr.py`, `run_ablation_study.py`,
`plot_roc_matched_fpr.py`, `gen_matched_fpr.py`, and the `JailBreakV-28K`
evaluator (a benchmark not used in the paper). Ad-hoc sanity probes
(`check_*.py`, `inspect_*.py`) are also omitted.

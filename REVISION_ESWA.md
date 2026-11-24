# Revision ESWA-D-26-42849 — new runs and analyses

This branch (`revision-eswa`) adds the experiments and analyses requested by the
three reviewers. Nothing in the published campaign changes: with the default
options every trajectory is bit-identical to `results/campaign` (verified on the
synthetic grid, LR/RF/GBDT).

## Engine changes (backward compatible)

| change | where | default = published behaviour |
|---|---|---|
| `learner_seed_offset` per strategy entry (Margin replicates) | `config.normalize_strategies`, `engine/runner.py`, `engine/trajectory.py` | `0` |
| independent evaluation set with a fixed number of rare cases (`synthetic_grid.fixed_test`) logged as `fx_auprc` | `engine/runner.fixed_test_set`, `engine/trajectory.py` | absent |
| `n_test`, `n_test_pos` logged every round | `engine/trajectory.py` | new columns only |
| learner variants `rf_leaf5`, `rf_trees50`, `gbdt_leaf5` (plus existing `rf_leaf1_500`, `*_unweighted`) | `learners.py` | unused |
| sharded parallel oracle (`oracle.n_jobs > 1`), any learner list | `experiments/oracle_parallel.py`, `cli.py` | sequential |
| campaigns on the oracle states (`--only oracle_campaign`) | `engine/states.oracle_campaign_states`, `cli.py` | not run |
| `"__replace__": true` in a config dict replaces instead of merging | `config._merge` | merge |
| Committee (LR members) is a cross-learner control (family `control`), excluded from the meta-regression families | `strategies/registry.py`, `analysis/metareg.py` | — |
| `role()` of every policy: published / reference / control / instance / probe | `strategies/registry.py` | — |
| learners `logreg_lbfgs`, `logreg_lbfgs_unweighted` (unpenalized intercept) and `logreg_iscale100` (liblinear, intercept penalty / 10^4) | `learners.py` | unused; `logreg` unchanged |
| threshold diagnostics `pool_share_p05`, `pool_prev` (`log_threshold_diagnostics: true`) | `engine/trajectory.py`, `engine/runner.py` | off (acquisition unchanged when on) |
| `rare-al swap-learner`: results tree with `logreg` rows replaced by `logreg_lbfgs` rows | `tools/swap_learner.py`, `cli.py` | — |

## New runs (configs/eswa/rev_*.json)

| config | reviewer point | what | approx. cost |
|---|---|---|---|
| `rev_noise_quick.json` | R3.1 | 3 Margin replicates (random ties + shifted learner seed), RF and GBDT, IR ∈ {20, 50, 100} | ~6 core-h |
| `rev_synthetic.json` | R3.1, R3.3 | full grid, all 27 strategies + 5 Margin replicates, extra evaluation set with 100 rare cases | ~130 core-h |
| `rev_noise_real.json` | R3.1, R3.5, R3.6 | 3 Margin replicates per real-data state | ~40 core-h |
| `rev_oracle.json` | R2.6, R2.8, R3.4 | oracle under LR/RF/GBDT on 288 base states (sharded) + campaigns on the same states | ~45 core-h |
| `rev_sensitivity.json` | R2.4, R1.3 | leaf size, number of trees, finer probability grid, no class weighting; 15 strategies; real (5 seeds) + high-rarity synthetic | ~60 core-h |
| `rev_lbfgs.json` | LR intercept (all LR results) | `logreg_lbfgs` on synthetic, rarity, real, image; 27 strategies + 5 Margin replicates; fixed test; threshold diagnostics | ~35 core-h |
| `rev_threshold.json` | LR intercept (mechanism) | Margin/Random/DIRECT with diagnostics under `logreg`, `logreg_iscale100`, RF, GBDT; real, image, synthetic IR ≥ 20 | ~25 core-h |
| `rev_budget.json` | R3 (budget claim) | n_init = 10, batch = 4, 21 rounds: 26 and 94 labels from the same trajectories | ~10 core-h |

```bash
pip install -e .
# reuse the published LR oracle states (only RF/GBDT and the new seeds are computed)
mkdir -p results/revision/oracle && cp results/campaign/oracle/oracle_*.csv results/revision/oracle/

# LR intercept first: every logistic-regression result depends on it
rare-al run --config configs/eswa/rev_lbfgs.json       --only real image        # ~1.5 h on 12 workers
rare-al run --config configs/eswa/rev_lbfgs.json       --only synthetic rarity  # ~1.5 h
rare-al run --config configs/eswa/rev_threshold.json   --only real image synthetic
rare-al run --config configs/eswa/rev_oracle_lbfgs.json --only oracle oracle_campaign observability   # LR oracle, ~1-2 h
rare-al swap-learner --base results/campaign --add results/revision/lbfgs --out results/campaign_lbfgs
rare-al analyze --config configs/eswa/campaign_lbfgs.json                       # every main table, LR = lbfgs

rare-al run --config configs/eswa/rev_noise_quick.json --only synthetic        # ~30 min on 12 workers
rare-al run --config configs/eswa/rev_oracle.json      --only oracle oracle_campaign
rare-al run --config configs/eswa/rev_budget.json      --only synthetic
rare-al run --config configs/eswa/rev_sensitivity.json --only real synthetic
rare-al run --config configs/eswa/rev_noise_real.json  --only real
rare-al run --config configs/eswa/rev_synthetic.json   --only synthetic         # longest
```

All runs are resumable (completed keys are skipped).

## Analyses (`rare-al revision-report`)

```bash
rare-al revision-report --config configs/eswa/rev_synthetic.json \
    --campaign results/campaign --out results/revision/assets
```

Each section runs when its inputs exist and is otherwise logged as missing
(`results/revision/assets/revision_report.md`). Outputs: `stats/rev_*.csv`,
`tables/rev_*.tex`, `figures/*.pdf`.

| section | reviewer point | function (`analysis/revision.py`) | inputs |
|---|---|---|---|
| noise floor (tie-only and replicate), excess tail, regret vs. replicate mean | R3.1 | `noise_floor`, `excess_tail` | rev_synthetic or rev_noise_quick |
| worst-case anatomy, lower quantiles with intervals | R3.2 | `worst_case_anatomy`, `lower_quantiles` | campaign synthetic |
| rare test cases per IR, rarity effect with 100 test positives | R3.3 | `test_composition`, `fixed_test_paired` | campaign / rev_synthetic |
| RQ3 with base-state unit, Holm/BH, TOST, conditional level; query- vs. campaign-level on the same states | R3.4, R2.6, R2.7 | `rq3_inference`, `query_vs_campaign` | oracle (+ oracle_campaign) |
| tolerance sweep: does the tail constraint change decisions and held-out losses? | R3.5, R2.2 | `validation_alpha_sweep` | campaign real + synthetic |
| pilot: runs needed; pilot simulation on held-out seeds | R3.6 | `pilot_sample_size`, `pilot_simulation` | campaign real |
| datasets and independent sources per stratum / domain | R3.7 | `source_group`, `strata_counts` | campaign real |
| budget at matched initial set and batch | R3 (budget) | `budget_at_rounds` | campaign synthetic, rev_budget |
| learner-setting sensitivity | R2.4, R1.3 | `sensitivity_summary` | rev_sensitivity |
| LR intercept: gains vs. Margin with liblinear vs. lbfgs on the same states, Margin noise floor and excess tail under lbfgs, threshold diagnostics | LR intercept | `intercept_effect`, `threshold_diagnostics` | campaign + rev_lbfgs, rev_threshold |
| Fig. 3 (zoom, contiguous numbering), Fig. 8 (two panels) | R1.9, R1.10, R2.12 | `reporting/revision_figures.py` | campaign synthetic, image |

## Logistic-regression intercept (found after the first revision pass)

`logreg` (all published LR results) is `LogisticRegression(solver="liblinear",
class_weight="balanced", C=1)`. liblinear appends a constant feature
(`intercept_scaling=1`) and L2-penalizes its weight like any other: with 30-190
labels the intercept stays near zero, the 0.5 threshold falls inside the majority
mass and Margin queries rare cases at about the chance rate. `logreg_lbfgs` is the
same model with an unpenalized intercept; `logreg_iscale100` keeps liblinear and
divides the intercept penalty by 10^4, which reproduces `logreg_lbfgs` (mechanism
control: the effect is the intercept penalty, not the solver). Tree learners are
unaffected. The v1 `committee` control keeps liblinear members (verbatim port);
under `logreg_lbfgs`, `committee_learner` is the committee with matched LR members.

`results/campaign_lbfgs` (from `swap-learner`) is `results/campaign` with the LR
rows replaced by `logreg_lbfgs` rows under the name `logreg` (column
`learner_impl` keeps the provenance), so `rare-al analyze` rebuilds every main
table and figure unchanged. `swap-learner` refuses an incomplete lbfgs run.

The revision analyses with LR = lbfgs need the same substitution in the revision
tree, keeping the Margin replicates (otherwise lbfgs strategies would be charged
the liblinear anchor's noise):

```bash
# 1. liblinear (as published) + the intercept comparison (tab_intercept.tex)
rare-al revision-report --config configs/eswa/rev_synthetic.json --campaign results/campaign \
    --out results/revision/assets
# 2. every revision analysis with LR = lbfgs
rare-al swap-learner --base results/revision --add results/revision/lbfgs --out results/revision_lbfgs \
    --keep-replicates --sections synthetic noise_real/real:real
rare-al swap-learner --base results/revision --add results/revision --out results/revision_lbfgs \
    --sections oracle oracle_campaign --no-clean        # oracle files already hold both LR variants
rare-al revision-report --config configs/eswa/rev_report_lbfgs.json --campaign results/campaign_lbfgs \
    --out results/revision_lbfgs/assets
rare-al validate-guidelines --config configs/eswa/campaign_lbfgs.json
```

Budget and sensitivity sections of the second report read `results/revision_lbfgs/{budget,sensitivity}`;
copy them from `results/revision` (their LR rows are liblinear: read them as such, or rerun
`rev_budget.json` / `rev_sensitivity.json` with `"learners": ["logreg_lbfgs", ...]`).


## Figures of the revised manuscript (`rare-al paper-figures`)

All figures of the revised manuscript and supplement are drawn at the printed text width (5.4 in) by one
command, which first writes the small CSV files they are drawn from (`<out>/figure_data/`) and then the PDFs:

```bash
rare-al paper-figures --campaign results/campaign --lbfgs results/campaign_lbfgs \
    --revision results/revision_lbfgs/assets --oracle results/revision/oracle --out paper/figures
rare-al paper-figures --out paper/figures --plot-only      # redraw from figure_data only
```

| figure | file | figure data | source |
|---|---|---|---|
| Fig. 2 same mean, different tails | `fig_headline_tail.pdf` | `synthetic_risk_profile.csv` | `analyze` on campaign_lbfgs |
| Fig. 3 rarity and dispersion | `fig_c1_v52.pdf` | `c1_decomposition.csv` | `analyze` |
| Fig. 4 configuration effect | `fig_config_effect.pdf` | `config_effect.csv`, `config_yield.csv` | campaign (liblinear) + campaign_lbfgs real/image |
| Fig. 5 real-data forest | `fig_forest_real.pdf` | `real_dataset_unit.csv` | `analyze` |
| Fig. 6 RQ3 | `fig_rq3.pdf` | `opportunity_by_k.csv`, `opportunity_diffs.csv`, `rev_query_vs_campaign.csv` | oracle points; `revision-report` |
| Fig. 7 validation by learner | `fig_validation.pdf` | `validation_by_learner.csv` | `revision-report` (validation cells) |
| Supplementary figures | `fig_mean_tail_rf_v52.pdf`, `fig_tail_vs_ir_rf.pdf`, `fig_heatmap.pdf`, `fig_multistep.pdf`, `fig_c5_v52.pdf`, `fig_image_backbones.pdf` | as above, `tail_by_ir.csv`, `domain_risk.csv`, `c5.csv`, `image_backbones.csv` | `analyze`, campaign_lbfgs |

The critical-difference diagrams and the risk map are those written by `rare-al analyze`
(`results/campaign_lbfgs/assets/figures`). Intervals: datasets clustered by the 31 independent sources
(tabular), datasets or tasks (images), base states (oracle), held-out units clustered by source or cell
(validation); seeds are fixed in `reporting/paper_figures.py`.

## Helper scripts (`scripts/revision/`)

* `extract_final.py <results_dir>` keeps the last round of every trajectory (compact transfer of large runs).
* `pool_share_round0.py` recomputes, without any campaign, the share of the pool scored at or above 0.5 by
  the model fitted on the 30 seed labels, for LR with a penalized intercept, an unpenalized intercept and the
  mechanism control, on every real and image state (Table 3 of the revised manuscript).

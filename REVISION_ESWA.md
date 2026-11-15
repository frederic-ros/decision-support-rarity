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

## New runs (configs/eswa/rev_*.json)

| config | reviewer point | what | approx. cost |
|---|---|---|---|
| `rev_noise_quick.json` | R3.1 | 3 Margin replicates (random ties + shifted learner seed), RF and GBDT, IR ∈ {20, 50, 100} | ~6 core-h |
| `rev_synthetic.json` | R3.1, R3.3 | full grid, all 27 strategies + 5 Margin replicates, extra evaluation set with 100 rare cases | ~130 core-h |
| `rev_noise_real.json` | R3.1, R3.5, R3.6 | 3 Margin replicates per real-data state | ~40 core-h |
| `rev_oracle.json` | R2.6, R2.8, R3.4 | oracle under LR/RF/GBDT on 288 base states (sharded) + campaigns on the same states | ~45 core-h |
| `rev_sensitivity.json` | R2.4, R1.3 | leaf size, number of trees, finer probability grid, no class weighting; 15 strategies; real (5 seeds) + high-rarity synthetic | ~60 core-h |
| `rev_budget.json` | R3 (budget claim) | n_init = 10, batch = 4, 21 rounds: 26 and 94 labels from the same trajectories | ~10 core-h |

```bash
pip install -e .
# reuse the published LR oracle states (only RF/GBDT and the new seeds are computed)
mkdir -p results/revision/oracle && cp results/campaign/oracle/oracle_*.csv results/revision/oracle/

rare-al run --config configs/eswa/rev_noise_quick.json --only synthetic        # first, ~30 min on 12 workers
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
| Fig. 3 (zoom, contiguous numbering), Fig. 8 (two panels) | R1.9, R1.10, R2.12 | `reporting/revision_figures.py` | campaign synthetic, image |

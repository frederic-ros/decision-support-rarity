# Changelog

## 2.1.0 — revision of ESWA-D-26-42849

Analyses and runs requested in the first review round, all resumable and documented in
[`REVISION_ESWA.md`](REVISION_ESWA.md). The published trajectories (v2.0.0) are reproduced bit for bit
with the default options.

* **Run-to-run noise floor** of the Margin anchor: random tie-breaking (`margin_rtb`) and Margin replicates
  with a shifted learner seed (`learner_seed_offset`); excess tail of every strategy over the floor
  (`rev_noise_quick`, `rev_synthetic`, `rev_noise_real`).
* **Fixed-positive evaluation set** (100 rare cases at the realized IR), acquisition unchanged (`fixed_test`).
* **Worst-case anatomy**, lower quantiles with intervals, rare test cases per IR.
* **RQ3 inference** with the base state as unit, Holm and Benjamini–Hochberg adjustment, TOST equivalence,
  conditional level; query-level vs. campaign-level comparison on the same oracle states (`oracle_campaign`);
  sharded multi-learner oracle (LR, RF, GBDT).
* **Out-of-sample validation** of the selection rules over a tolerance sweep, the anchor charged its own noise;
  **pilot** sample size and simulation on held-out seeds.
* **Logistic-regression configuration**: `logreg_lbfgs` (unpenalized intercept), `logreg_iscale100`
  (mechanism control), threshold diagnostics (`log_threshold_diagnostics`); `rare-al swap-learner` rebuilds
  every table with the corrected linear learner.
* **Learner-setting sensitivity** (leaf size, number of trees, probability resolution, class weighting).
* **Matched low-budget run** (26 and 94 labels from the same trajectories).
* **`rare-al paper-figures`**: every figure of the revised manuscript at the printed text width, with the
  figure data written alongside; includes the discovery-mechanism figure (material-loss probability by
  rare-label gap to the anchor, for the policies and for the anchor's tie-breaking rerun).
* `rare-al revision-report`: all revision tables and statistics.

## 2.0.0 — submitted version

Paired-trajectory protocol, 27 acquisition rules, three learners, synthetic grid, 42 real datasets,
image benchmark on two embeddings, discovery oracle and observability probe.

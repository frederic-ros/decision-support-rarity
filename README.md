# decision-support-rarity — Risk-aware active learning under class rarity

Companion software for the paper
*"Risk-aware active learning under class rarity: A decision framework and
cross-domain evaluation for expert systems"* (Expert Systems with Applications).

It evaluates **27 acquisition strategies** against a **margin** reference under a
strictly **paired-trajectory** protocol, across a synthetic grid, a rarity-stress
sweep, a cross-domain real benchmark, an **image benchmark with two embeddings**,
and a discovery-oracle / observability probe. One script regenerates every table
and figure in the paper from the archived results.

- **Paper:** doi:10.1016/j.eswa.2026.XXXXXX
- **Data & results archive (Zenodo):** doi:10.5281/zenodo.XXXXXXX
- **License:** MIT (see [`LICENSE`](LICENSE)); bundled benchmark datasets keep their source licenses.

> This release consolidates and extends our prior active-learning engine: the
> paired protocol, the acquisition functions, the oracle and the statistical
> analysis are reused verbatim (a frozen port lives under `strategies/v1/`), with
> the risk-aware framework wired around them.

## What the framework adds

- **Rarity preserved, not compressed.** Stratified, prevalence-preserving
  subsampling keeps at least `min_rare` rare cases, so the evaluated imbalance
  ratio matches the nominal one (the earlier size cap that compressed IR — e.g.
  `protein_homo` 111→5.2 — is gone; `legacy_cap` is kept only for the audit).
- **Image datasets with two embeddings.** Each image set becomes frozen
  embeddings from **two backbones of contrasting supervision** — DINOv2 ViT-S/14
  (self-supervised, 384-d) and ResNet-18 (ImageNet-supervised, 512-d) — with rare
  tasks built by `natural` rarity (e.g. ISIC melanoma) or `one_vs_rest`
  controllable IR (CIFAR / GTSRB / EuroSAT / …). Image tasks flow through the
  identical engine, metrics and analysis as the tabular benchmark.
- **12 new strategies** on top of 15 earlier ones (see below), with a stateful
  `Strategy`/`Context` interface so adaptive bandits (TAILOR, ALBL) and
  rare-targeted methods (SIMILAR-FLQMI, GALAXY, DIRECT, AnchorAL, class-balanced,
  RCA) keep state across rounds.
- **GBDT learner** (`HistGradientBoosting`) added to `logreg` / `rf`.

## The 27 strategies

`margin` (reference), `margin_rtb`, `entropy`, `least_confident`, `random`,
`robust_uncertainty`, `committee`, `committee_learner`, `coreset`,
`kmeans_diversity`, `typiclust`, `probcover`, `uherding`, `dcf`,
`constrained_cover`, `rareguard`, `adaptive_repulsion`, `most_likely_positive`,
`ens`, `class_balanced`, `anchoral`, `similar_flqmi`, `galaxy`, `direct`,
`tailor`, `albl`, `rca`. Family / status / citation metadata lives in
`rare_al.strategies.registry.METADATA`.

## Install

```bash
bash runners/00_install.sh          # CPU core (pip install -e .)
pip install -e ".[images]"          # torch + torchvision — only to EXTRACT embeddings (GPU)
pip install -e ".[boost]"           # LightGBM / XGBoost learners (optional)
```

Exact environment used for the reported run: Python 3.11.17, scikit-learn 1.9.1,
numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, statsmodels 0.15.0, matplotlib 3.11.2
(pinned in [`requirements-pinned.txt`](requirements-pinned.txt)).

## Data and results

`data/` and `results/` are **not** tracked in this repository (they are large and
regenerable). Two ways to get them:

1. **Download the archive** from Zenodo (doi:10.5281/zenodo.XXXXXXX) and unzip the
   derived data into `data/` and the result records into `results/`. The archive
   bundles the image **embedding caches**, the moderate strata (`rare27`,
   `standard20`) and the run outputs (`campaign`, `first_pass`). See the archive's
   `ARCHIVE_README.md`.
2. **Fetch and regenerate** from the original providers:

```bash
bash runners/check_data.sh                                     # read-only: what's present vs missing
bash runners/01b_fetch_domains.sh                              # auto-fetch the cross-domain datasets
bash runners/01_download_data.sh      configs/eswa/full.json   # tabular + domain cache + manifest
bash runners/02_extract_images.sh     configs/eswa/full.json   # 2 backbones per image set (GPU + network)
bash runners/10_run_all.sh            configs/eswa/full.json   # synthetic, rarity, real, image, oracle, observability
bash runners/20_analyze.sh            configs/eswa/full.json   # tables + figures -> results/.../assets
```

Raw image sets download automatically via `torchvision`; cross-domain datasets are
pulled by `01b_fetch_domains.sh` (UCI automatic; Kaggle via the `kaggle` CLI;
CIC-IDS2017 / ISIC manual). See [`DATA_README.md`](DATA_README.md) for the exact
layout and per-dataset sources. Windows `.bat` equivalents sit beside each script.

**Fast offline check** (no network, no GPU, ~2 min, fabricated fixtures):

```bash
bash runners/03_smoke.sh
```

## Regenerate the paper's tables and figures

If you downloaded the Zenodo archive (or ran the campaign yourself):

```bash
rare-al analyze --config configs/eswa/campaign.json
# outputs land in results/campaign/assets/{tables,figures,stats}
```

## CLI

```
rare-al prepare-data        --config CONFIG [--raw-dir DIR] [--rare27-dir DIR] [--force]
rare-al extract-images      --config CONFIG [--out DIR] [--data-root DIR] [--force]
rare-al run                 --config CONFIG [--only synthetic rarity real image oracle observability]
rare-al analyze             --config CONFIG [--out DIR] [--fast]
rare-al validate-guidelines --config CONFIG [--tau 0.10] [--k 5]   # leave-one-domain-out, post-campaign
```

`validate-guidelines` is the out-of-sample test of the decision procedure: it fits
the strategy-selection rule on all domains but one, applies it to each held-out
dataset using only observable (initial-label) descriptors, and compares the
realized held-out gain against always-Margin and the mean-rank pick (writes
`tab_validation.tex`).

Runs are **resumable** (results append to `trajectories.csv`; completed keys are
skipped) and the analyzer is **fault-tolerant** (a degenerate section is logged
and skipped, so a partial run still yields a partial report).

## Config

`configs/eswa/full.json` — all 27 strategies × {logreg, rf, gbdt} × two budgets,
the full real benchmark, the image benchmark (image sets × 2 backbones × positive
classes × target IRs × seeds), the synthetic grid, the rarity sweep and the
oracle. `configs/eswa/campaign.json` is the reported configuration;
`configs/smoke/smoke.json` is a 2-minute subset. A config may `extends` another.

## Layout

```
src/rare_al/
  cli.py                 prepare-data / extract-images / run / analyze
  config.py io.py repro.py learners.py
  data/      splits subsample synthetic images loaders registry prepare complexity
  strategies/ base basic active_search rare_targeted adaptive rca registry
              v1/  (frozen verbatim port of the 15 earlier acquisition functions)
  engine/    states trajectory runner
  metrics/   endpoints            (AUPRC, recall, R@capacity, cost, AUBC, labels-to-k)
  experiments/ oracle observability
  analysis/  load paired risk oracle descriptors metareg c5
  reporting/ tables figures assets
configs/   eswa/{full,campaign}.json  smoke/smoke.json
runners/   00..20 (.sh + .bat)
tests/     test_smoke.py  make_smoke_fixtures.py
```

## Citation

If you use this software, please cite the paper and the archived release.

```bibtex
@article{ros2026riskaware,
  title   = {Risk-aware active learning under class rarity: A decision framework
             and cross-domain evaluation for expert systems},
  author  = {Ros, Fr\'ed\'eric and Bertrand-Bonvillain, Erwan and Nasser, Yassine},
  journal = {Expert Systems with Applications},
  year    = {2026},
  doi     = {10.1016/j.eswa.2026.XXXXXX}
}

@software{ros2026decisionsupportrarity,
  title     = {decision-support-rarity: Risk-aware active learning under class rarity},
  author    = {Ros, Fr\'ed\'eric and Bertrand-Bonvillain, Erwan and Nasser, Yassine},
  version   = {2.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/frederic-ros/decision-support-rarity}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

MIT — see [`LICENSE`](LICENSE). Bundled benchmark datasets retain their source
licenses; see [`DATA_README.md`](DATA_README.md) and the Zenodo archive's
`ARCHIVE_README.md`.

# Software lineage

This package (`rare_al`, distribution `rare-al`, v2.0.0) is the implementation
accompanying the paper *"Risk-aware active learning under class rarity: A
decision framework and cross-domain evaluation for expert systems."* It builds on
the authors' earlier active-learning engine and extends it with the risk-aware
framework evaluated in the paper.

## Frozen core

`src/rare_al/strategies/v1/` is a frozen port of an earlier set of acquisition
functions (`common, uncertainty, geometry, coverage, rareaware, committee`),
reused verbatim so that prior results remain reproducible. These modules are kept
stable and should not be edited.

## Added in this release

- **Preserved rarity.** `data/subsample.py` performs proportional,
  prevalence-preserving stratified subsampling, so the evaluated imbalance ratio
  matches the nominal one (a non-preserving size cap, retained only for audit as
  `legacy_cap`, would compress it — e.g. `protein_homo` 111→≈5).
- **Image benchmark with two embeddings.** `data/images.py` produces frozen
  embeddings from two backbones of contrasting supervision (DINOv2 ViT-S/14,
  ResNet-18); `engine/states.py` and `engine/runner.py` make image tasks
  first-class states that flow through the identical runner, metrics and analysis
  as the tabular benchmark, carrying `backbone` / `positive_class` / `ir_target`
  into every output row.
- **Twelve additional strategies** with a stateful `Strategy`/`Context` interface
  (adaptive bandits and rare-targeted methods keep state across rounds), the GBDT
  learner, the discovery-oracle / observability experiments, the risk-aware
  reporting (`reporting/`), the `rare-al` CLI, configurations, runners, tests and
  documentation.

## Reproducibility

All randomness derives deterministically from a master seed (state, method and
learner seeds), so any state can be regenerated in isolation. `rare-al analyze`
regenerates every table and figure from the archived result records. The exact
package versions and hardware used for the reported run are listed in the paper's
reproducibility appendix and in `requirements-pinned.txt`. The image benchmark
requires real embeddings; the offline smoke fixtures use separable synthetic
blobs and are intended only to exercise the pipeline, not to produce meaningful
image numbers.

> **Availability.** `data/` and `results/` are not tracked in the git repository.
> The derived, redistributable data (image embedding caches, the `rare27` and
> `standard20` strata) and the result records are archived on Zenodo
> (doi:10.5281/zenodo.23292060); see that record's `ARCHIVE_README.md`. The raw
> third-party datasets below are obtained from their original providers with the
> scripts in this repository.

# Data setup

## Recommended `data/` layout

Everything lives under `data/` in four subfolders (the config points at these):

```
data/
├─ rare27/        27 xlsx (moderate stratum, offline)        -> data.rare27_dir
├─ standard20/    20 xlsx (balanced control, offline)        -> data.standard20_dir
├─ embeddings/    <dataset>/<backbone>.npz  (REUSED as-is)   -> image_benchmark.embedding_cache_dirs
├─ images/        raw image sets — only if you must RE-extract-> image_benchmark.data_root
└─ domain/        cross-domain datasets, one subfolder each   -> data.raw_dir
```

If you already have extracted embeddings laid out as
`data/embeddings/<dataset>/dinov2_vits14.npz` + `resnet18.npz` (keys X/y or
embeddings/labels — both accepted), **extraction is skipped entirely**: the engine
reads those directly. `data/images/` (raw torchvision folders) is only needed to
re-extract a set that has no embedding yet.

Check what you already have in place at any time (read-only):

```bash
python scripts/check_data.py --config configs/eswa/full.json
```

The tiers below say where each piece comes from.

## 1. Moderate stratum — 27 imbalanced-learn sets

Two ways, pick one:

- **Offline from a Rare27 xlsx export (recommended if you have it).** Point
  `data.rare27_dir` at the folder of `*.xlsx` (default `native/rare27`; columns
  `feature_*` + `target_binary`). `prepare-data` reads them directly — no
  network. Verified IR-preserving (e.g. `protein_homo` 111.5→111.4,
  `abalone_19` 129.5→129.5; big sets capped to `real_n_max` at constant IR).
- **Download via imbalanced-learn.** Leave `rare27_dir` unset and
  `prepare-data` fetches each set with `imblearn.datasets.fetch_datasets`
  (network the first time, cached afterwards).

Either way they are re-evaluated **without** the v13 IR compression.

## 1b. Cross-domain datasets — automatic fetch

`bash runners/01b_fetch_domains.sh` (or `python scripts/fetch_domains.py`) puts
each domain dataset under `data/raw/<folder>/` with the exact filenames the
loaders expect, then prints a checklist:

- **UCI** (taiwan / polish bankruptcy, ai4i2020, aps_failure, secom) — fully
  automatic, no account.
- **Kaggle** (creditcard, baf, unsw_nb15) — automatic **if** the `kaggle` CLI is
  installed and configured (`pip install kaggle`, `~/.kaggle/kaggle.json`);
  otherwise the script prints the one-click page.
- **Manual** (CIC-IDS2017 corrected LYCOS, ISIC-2020 embeddings) — the script
  prints the source and the exact target folder.

`python scripts/fetch_domains.py --list` shows the plan; `--only secom aps_failure`
fetches a subset.

## 2. Cross-domain real benchmark (drop raw files under `data/raw/`)

Put the raw files in `data/raw/<folder>/` as below, then run `prepare-data`.
Missing files are skipped with a message — you can prepare whatever subset you
have. Each prepared set is stratified-subsampled to preserve its nominal IR and
cached as `data/cache/<id>.npz` with provenance in `data/cache/manifest.json`.

| domain | dataset id(s) | `data/raw/<folder>/` | notes |
|---|---|---|---|
| finance | `creditcard` | `creditcard/` | Kaggle ULB credit-card fraud CSV |
| finance | `baf_base` | `baf/` | Bank-Account-Fraud (NeurIPS 2022) base CSV |
| finance | `taiwan_bankruptcy` | `taiwan_bankruptcy/` | UCI Taiwanese bankruptcy CSV |
| finance | `polish_bankruptcy_y1..y5` | `polish_bankruptcy/` | UCI `.arff`, one per year |
| industry | `ai4i2020` | `ai4i2020/` | UCI AI4I 2020 predictive-maintenance CSV |
| industry | `aps_failure` | `aps_failure/` | Scania APS train/test CSV |
| industry | `secom` | `secom/` | UCI SECOM (data + labels) |
| cyber | `unsw_nb15_{analysis,backdoor,shellcode,worms}` | `unsw_nb15/` | UNSW-NB15 train/test CSV (one-vs-benign per attack) |
| cyber | `cicids2017_{heartbleed,infiltration,sql_injection}` | `cicids2017/` | **use the corrected LYCOS-IDS2017** CSVs |
| medicine | `isic2020_emb` | `isic2020/` | pre-extracted embedding `.npz` (`X`, `y`) |

The exact filenames / columns are declared per dataset in
`src/rare_al/data/registry.py` (`DatasetSpec.params`); adjust there if your copy
differs. `adbench_*` and fully custom sets can be registered with
`registry.register_adbench` / `register_custom`.

## 3. Image benchmark — two embeddings

`rare-al extract-images` writes `data/image_cache/<dataset>/<backbone>.npz`
(`X` float32, `y` multiclass) for each image set × each backbone
(`dinov2_vits14`, `resnet18`). Needs a GPU, network and `pip install -e ".[images]"`.

- **torchvision sets download automatically**: `cifar10`, `cifar100`, `gtsrb`,
  `eurosat`, `stl10`, `svhn`, `pcam`, `fashion_mnist`.
- **local sets** (e.g. `isic2020`): provide an `ImageFolder` at
  `data/images_raw/<dataset>/<class>/*.jpg`.

Rarity is set in the config: `natural` (a genuinely rare class, e.g. ISIC
melanoma) or `one_vs_rest` (one class becomes the rare positive, subsampled to
`target_ir`).

**Reuse an existing cache.** If you already have embeddings laid out as
`<dir>/<dataset>/<backbone>.npz` (for instance a forecastability cache), add that
directory to `image_benchmark.embedding_cache_dirs` and skip extraction for those
sets. Both key conventions are accepted: `X`/`embeddings` and `y`/`labels`.

## Offline smoke fixtures

`python tests/make_smoke_fixtures.py` fabricates a tiny CIFAR-10 embedding cache
and a small prepared real cache so `configs/smoke/smoke.json` runs with no
network and no GPU. These are synthetic blobs, not real data.

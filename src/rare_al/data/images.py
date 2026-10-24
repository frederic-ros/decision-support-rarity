"""Image datasets as frozen embeddings, with controllable rarity.

Reviewer requirement: image datasets evaluated with two embeddings. The engine
consumes ``(X, y)`` arrays, so an image task is just an embedding matrix plus a
binary rare label -- it flows through the *same* paired-trajectory engine,
operational metrics and analysis as the tabular benchmark.

Embeddings are extracted ONCE per (dataset, backbone) with two backbones that
differ in supervision:

  * ``dinov2_vits14`` -- DINOv2 ViT-S/14, self-supervised, 384-d
  * ``resnet18``      -- ResNet-18, ImageNet-supervised, 512-d

They are cached as ``<dir>/<dataset>/<backbone>.npz`` (keys ``X`` and ``y``), then
reused to build many rare tasks cheaply. An existing cache laid out that way (for
instance a forecastability embedding cache) is reused as-is; extraction only runs
when the cache is absent and needs a GPU + network.

Two ways to obtain rarity:

  * ``natural``     -- a naturally rare class (ISIC 2020 melanoma ~1.8 %). The pool
                       is stratified-subsampled so the nominal prevalence is kept
                       and at least ``min_rare`` rare cases remain.
  * ``one_vs_rest`` -- one class of a multiclass set becomes the rare positive and
                       is subsampled to a target IR against the rest -- controllable
                       rarity under the identical protocol as the tabular study.

Building a rare task never needs a backbone, only the cached embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rare_al.data.subsample import stratified_subsample, target_size

# backbone id -> (embedding dim, loader kind). Extraction is lazy (user machine).
BACKBONES = {
    "dinov2_vits14": {"dim": 384, "kind": "dinov2"},
    "resnet18": {"dim": 512, "kind": "resnet18"},
}


@dataclass
class ImageTaskSpec:
    dataset: str                 # cifar10, cifar100, gtsrb, eurosat, isic2020, pcam...
    backbone: str                # dinov2_vits14 | resnet18
    rarity: str = "one_vs_rest"  # one_vs_rest | natural
    positive_class: int | str = 0
    target_ir: float = 99.0      # ~1% prevalence; ignored when rarity == "natural"
    min_rare: int = 50
    max_pool: int = 10_000
    n_hard_max: int = 60_000


# ---------------------------------------------------------------- embeddings
def embedding_path(cache_dir, dataset, backbone) -> Path:
    return Path(cache_dir) / dataset / f"{backbone}.npz"


def _xy_from_npz(z, path):
    """Extract (X, y) from an .npz with flexible key names. Falls back to shape:
    the widest 2-D array is X, a matching-length 1-D array is y."""
    xkeys = ("X", "embeddings", "features", "feats", "Z", "emb")
    ykeys = ("y", "labels", "label", "targets", "target", "Y")
    X = next((z[k] for k in xkeys if k in z.files), None)
    y = next((z[k] for k in ykeys if k in z.files), None)
    if X is None:
        cand = [k for k in z.files if getattr(z[k], "ndim", 0) == 2]
        if not cand:
            raise KeyError(f"{path}: no 2-D array found; keys={z.files}")
        X = z[max(cand, key=lambda k: z[k].shape[1])]
    if y is None:
        n = len(X)
        cand = [k for k in z.files if getattr(z[k], "ndim", 0) == 1 and len(z[k]) == n]
        if not cand:
            raise KeyError(f"{path}: no label vector of length {n}; keys={z.files}")
        y = z[cand[0]]
    return np.asarray(X, float), np.asarray(y).ravel()


def load_multiclass_embeddings(dataset, backbone, cache_dirs):
    """Load cached ``(X, y_multiclass)`` from the first of ``cache_dirs`` holding
    ``<dir>/<dataset>/<backbone>.npz``. Key names are detected flexibly
    (X/embeddings/features, y/labels/targets, else by array shape), so an existing
    forecastability-style cache is reused unchanged."""
    tried = []
    for d in cache_dirs:
        p = embedding_path(d, dataset, backbone)
        tried.append(str(p))
        if p.exists():
            return _xy_from_npz(np.load(p, allow_pickle=True), p)
    raise FileNotFoundError(
        f"no cached embeddings for {dataset}/{backbone}; looked in {tried}. "
        f"Run `rare-al extract-images` on a machine with a GPU + network.")


def extract_embeddings(dataset, backbone, out_dir, *, data_root="data/images_raw",
                       batch_size=256, device=None):
    """Extract embeddings with ``backbone`` and cache ``(X, y_multiclass)``.

    Heavy imports are lazy: torch / torchvision are only needed on the extraction
    machine. torchvision multiclass sets (cifar10/100, fashion_mnist, gtsrb,
    eurosat, stl10, svhn, pcam) are downloaded automatically; ISIC and other local
    sets are read from an ImageFolder at ``data_root/<dataset>``.
    """
    import torch
    from torch.utils.data import DataLoader
    import torchvision as tv
    from torchvision import transforms as T

    if backbone not in BACKBONES:
        raise KeyError(f"unknown backbone {backbone!r}; known: {list(BACKBONES)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = embedding_path(out_dir, dataset, backbone)
    out.parent.mkdir(parents=True, exist_ok=True)

    if BACKBONES[backbone]["kind"] == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
        embed = lambda x: model(x)                                   # [B, 384]
    else:  # resnet18 penultimate (512-d)
        net = tv.models.resnet18(weights=tv.models.ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        net.fc = torch.nn.Identity()
        embed = lambda x: net(x)                                     # [B, 512]

    size, mean, std = 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    # convert("RGB") promotes 1-channel (MNIST family) and 4-channel (Caltech) alike
    tf = T.Compose([T.Lambda(lambda im: im.convert("RGB")),
                    T.Resize(size), T.CenterCrop(size),
                    T.ToTensor(), T.Normalize(mean, std)])
    ds = _torchvision_dataset(dataset, data_root, tf, tv)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)

    Xs, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            Xs.append(embed(xb.to(device)).cpu().numpy())
            ys.append(np.asarray(yb))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    np.savez_compressed(out, X=X.astype(np.float32), y=y)
    return out, X.shape


def _torchvision_dataset(dataset, root, tf, tv, download=True):
    """Map a dataset key to a torchvision dataset. ``download=True`` makes
    torchvision reuse already-extracted folders under ``root`` (it verifies the
    integrity of what is there and only fetches what is missing), so pointing
    ``root`` at a folder that already holds cifar-10-batches-py, gtsrb, eurosat,
    food-101, ... extracts entirely offline. Keys accept a few aliases so the
    config can use either 'flowers102' or 'flowers-102'."""
    R = dict(root=root, transform=tf, download=download)
    key = dataset.lower().replace("-", "_").replace(" ", "_")
    aliases = {"flowers_102": "flowers102", "food_101": "food101",
               "fgvc_aircraft_2013b": "fgvc_aircraft", "oxford_iiit_pet": "oxford_pet",
               "cifar_10_batches_py": "cifar10", "cifar_100_python": "cifar100",
               "stl10_binary": "stl10", "fashionmnist": "fashion_mnist",
               "emnist_balanced": "emnist"}
    key = aliases.get(key, key)

    def concat(*ctors):
        from torch.utils.data import ConcatDataset
        parts = []
        for c in ctors:
            try:
                parts.append(c())
            except Exception as exc:            # a split missing on disk is fine
                print(f"    [{key}] split skipped: {exc}", flush=True)
        if not parts:
            raise FileNotFoundError(f"no usable split for {key} under {root}")
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    # Fine-grained sets have tiny train splits, so concatenate every split to
    # maximise images per class (labels are consistent across splits).
    m = {
        "cifar10": lambda: concat(lambda: tv.datasets.CIFAR10(train=True, **R),
                                  lambda: tv.datasets.CIFAR10(train=False, **R)),
        "cifar100": lambda: concat(lambda: tv.datasets.CIFAR100(train=True, **R),
                                   lambda: tv.datasets.CIFAR100(train=False, **R)),
        "mnist": lambda: tv.datasets.MNIST(train=True, **R),
        "fashion_mnist": lambda: tv.datasets.FashionMNIST(train=True, **R),
        "kmnist": lambda: tv.datasets.KMNIST(train=True, **R),
        "emnist": lambda: tv.datasets.EMNIST(split="balanced", train=True, **R),
        "gtsrb": lambda: concat(lambda: tv.datasets.GTSRB(split="train", **R),
                                lambda: tv.datasets.GTSRB(split="test", **R)),
        "eurosat": lambda: tv.datasets.EuroSAT(**R),
        "stl10": lambda: concat(lambda: tv.datasets.STL10(split="train", **R),
                                lambda: tv.datasets.STL10(split="test", **R)),
        "svhn": lambda: concat(lambda: tv.datasets.SVHN(split="train", **R),
                               lambda: tv.datasets.SVHN(split="test", **R)),
        "pcam": lambda: tv.datasets.PCAM(split="train", **R),
        "caltech101": lambda: tv.datasets.Caltech101(target_type="category", **R),
        "dtd": lambda: concat(lambda: tv.datasets.DTD(split="train", **R),
                              lambda: tv.datasets.DTD(split="val", **R),
                              lambda: tv.datasets.DTD(split="test", **R)),
        "flowers102": lambda: concat(lambda: tv.datasets.Flowers102(split="train", **R),
                                     lambda: tv.datasets.Flowers102(split="val", **R),
                                     lambda: tv.datasets.Flowers102(split="test", **R)),
        "food101": lambda: concat(lambda: tv.datasets.Food101(split="train", **R),
                                  lambda: tv.datasets.Food101(split="test", **R)),
        "fgvc_aircraft": lambda: concat(lambda: tv.datasets.FGVCAircraft(split="trainval", **R),
                                        lambda: tv.datasets.FGVCAircraft(split="test", **R)),
        "oxford_pet": lambda: concat(lambda: tv.datasets.OxfordIIITPet(split="trainval", **R),
                                     lambda: tv.datasets.OxfordIIITPet(split="test", **R)),
    }
    if key in m:
        return m[key]()
    from torchvision.datasets import ImageFolder   # ISIC and other local sets
    return ImageFolder(str(Path(root) / dataset), transform=tf)


# ---------------------------------------------------------------- rare task
def build_rare_task(X, y_multiclass, spec: ImageTaskSpec, rng=None):
    """Return ``(X_task, y_binary, meta)`` with rare == 1 and the prevalence
    controlled by ``spec``. Deterministic given ``rng``."""
    rng = rng if rng is not None else np.random.default_rng(0)
    seed = int(rng.integers(0, 2**31 - 1))
    y_multiclass = np.asarray(y_multiclass)
    y_bin_full = (y_multiclass.astype(str) == str(spec.positive_class)).astype(int)

    if spec.rarity == "natural":
        n, n_rare = len(y_bin_full), int(y_bin_full.sum())
        size = target_size(n, n_rare, spec.max_pool, spec.min_rare, spec.n_hard_max)
        keep = stratified_subsample(y_bin_full, size, seed)
        idx = np.sort(keep)
    else:  # one_vs_rest: cap the majority, then subsample the positive to target_ir
        pos = np.flatnonzero(y_bin_full == 1)
        neg = np.flatnonzero(y_bin_full == 0)
        if len(neg) > spec.max_pool:
            neg = rng.choice(neg, spec.max_pool, replace=False)
        n_pos_target = max(spec.min_rare, int(round(len(neg) / spec.target_ir)))
        n_pos = min(n_pos_target, len(pos))
        pos = rng.choice(pos, n_pos, replace=False)
        idx = np.sort(np.concatenate([neg, pos]))

    Xt = np.asarray(X, float)[idx]
    yt = y_bin_full[idx]
    n_rare = int(yt.sum())
    n_major = int((yt == 0).sum())
    meta = {
        "dataset": spec.dataset, "backbone": spec.backbone, "rarity": spec.rarity,
        "positive_class": str(spec.positive_class),
        "ir_target": float(spec.target_ir) if spec.rarity == "one_vs_rest" else float("nan"),
        "ir_nominal": round(int((y_bin_full == 0).sum()) / max(int(y_bin_full.sum()), 1), 3),
        "ir_evaluated": round(n_major / max(n_rare, 1), 3),
        "n_pool": len(idx), "n_rare": n_rare, "d": Xt.shape[1],
        "min_rare_satisfied": bool(n_rare >= spec.min_rare),
    }
    return Xt, yt, meta


def make_task(spec: ImageTaskSpec, cache_dirs, rng=None):
    """Load cached embeddings and build the rare task ``(X, y, meta)``."""
    X, ymc = load_multiclass_embeddings(spec.dataset, spec.backbone, cache_dirs)
    return build_rare_task(X, ymc, spec, rng=rng)

#!/usr/bin/env python3
"""Read-only inventory of everything under data/: images, rare27, standard20 and
the cross-domain sets. Tells you what is in place and what is still missing,
before you extract embeddings or run prepare-data. Changes nothing.

    python scripts/check_data.py --config configs/eswa/full.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rare_al.config import load_config

# image dataset key -> candidate folder/file names under data_root (what
# torchvision looks for; the raw download layout)
IMAGE_LAYOUT = {
    "cifar10": ["cifar-10-batches-py"],
    "cifar100": ["cifar-100-python"],
    "mnist": ["MNIST"],
    "fashion_mnist": ["FashionMNIST"],
    "kmnist": ["KMNIST"],
    "emnist": ["EMNIST"],
    "gtsrb": ["gtsrb", "GTSRB"],
    "eurosat": ["eurosat", "2750", "EuroSAT"],
    "stl10": ["stl10_binary"],
    "svhn": ["train_32x32.mat", "test_32x32.mat"],
    "pcam": ["pcam", "PCAM", "camelyonpatch_level_2_split_train_x.h5"],
    "caltech101": ["caltech101", "101_ObjectCategories"],
    "dtd": ["dtd"],
    "flowers102": ["flowers-102"],
    "food101": ["food-101"],
    "fgvc_aircraft": ["fgvc-aircraft-2013b"],
    "oxford_pet": ["oxford-iiit-pet"],
    "oxford_iiit_pet": ["oxford-iiit-pet", "oxford_iiit_pet"],
    "emnist_balanced": ["EMNIST"],
}

RARE27 = ["abalone", "abalone_19", "arrhythmia", "car_eval_34", "car_eval_4",
          "coil_2000", "ecoli", "isolet", "letter_img", "libras_move", "mammography",
          "oil", "optical_digits", "ozone_level", "pen_digits", "protein_homo",
          "satimage", "scene", "sick_euthyroid", "solar_flare_m0", "spectrometer",
          "thyroid_sick", "us_crime", "webpage", "wine_quality", "yeast_me2", "yeast_ml8"]

STANDARD20 = ["balance-scale", "banknote-authentication", "breast-w", "cmc", "diabetes",
              "dna", "ionosphere", "iris", "mfeat-factors", "nursery", "phoneme",
              "qsar-biodeg", "segment", "sonar", "spambase", "tic-tac-toe", "vehicle",
              "vowel", "wall-robot-navigation", "wdbc"]


def _mark(ok):
    return "OK " if ok else ".. "


def check_images(cfg):
    img = cfg.get("image_benchmark", {})
    root = Path(img.get("data_root", "data/images"))
    cache_dirs = [Path(d) for d in img.get("embedding_cache_dirs", ["data/embeddings"])]
    backbones = img.get("backbones", ["dinov2_vits14", "resnet18"])
    print(f"\n== images  (embeddings = {cache_dirs[0]} ; raw = {root})")
    wanted = [ds["dataset"].lower().replace("-", "_") for ds in img.get("datasets", [])]
    ready = 0
    for k in dict.fromkeys(wanted):
        # embeddings are what we actually need: both backbones present?
        emb = None
        for d in cache_dirs:
            if all((d / k / f"{b}.npz").exists() for b in backbones):
                emb = d / k
                break
        raw_hit = next((c for c in IMAGE_LAYOUT.get(k, [k]) if (root / c).exists()), None)
        ok = emb is not None or raw_hit is not None
        ready += ok
        if emb is not None:
            status = f"embeddings OK ({', '.join(backbones)})"
        elif raw_hit is not None:
            status = f"raw present ({raw_hit}) - needs extract-images"
        else:
            status = "MISSING (no embeddings, no raw)"
        print(f"   [{_mark(ok)}] {k:18s} {status}")
    print(f"   -> {ready}/{len(dict.fromkeys(wanted))} image sets usable "
          f"(embeddings preferred; raw only if you must re-extract)")


def check_xlsx_dir(title, folder, expected):
    folder = Path(folder)
    print(f"\n== {title}  ({folder})")
    if not folder.exists():
        print(f"   folder does not exist yet -> 0/{len(expected)}")
        return
    have = {p.stem for p in folder.glob("*.xlsx")}
    miss = [e for e in expected if e not in have]
    print(f"   {len(expected) - len(miss)}/{len(expected)} present"
          + (f"   missing: {', '.join(miss)}" if miss else ""))


def check_domain(cfg):
    raw = Path(cfg["data"].get("raw_dir", "data/domain"))
    print(f"\n== domain datasets  ({raw})")
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import fetch_domains as fd
    except Exception:
        print("   (fetch_domains not importable; skipping)")
        return
    present = 0
    for k, spec in fd.DATASETS.items():
        out = raw / spec["folder"]
        ok, miss = fd.verify(spec, out) if out.exists() else (False, spec["expect"])
        present += ok
        tail = "" if ok else f"  (missing: {', '.join(miss)})"
        print(f"   [{_mark(ok)}] {k:18s} {out}{tail}")
    print(f"   -> {present}/{len(fd.DATASETS)} domain sets ready")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/eswa/full.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    print("=" * 60)
    print(f"DATA INVENTORY  ({args.config})")
    print("=" * 60)
    check_images(cfg)
    check_xlsx_dir("rare27 (moderate stratum)", cfg["data"].get("rare27_dir", "data/rare27"), RARE27)
    s20 = cfg["data"].get("standard20_dir", "data/standard20")
    check_xlsx_dir("standard20 (balanced control)", s20, STANDARD20)
    check_domain(cfg)
    print("\nNothing was modified (read-only).")


if __name__ == "__main__":
    main()

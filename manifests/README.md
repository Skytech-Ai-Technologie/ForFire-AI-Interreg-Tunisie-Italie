# Manifests

These manifests make the assembled corpus auditable and reproducible. The
two CSVs here ship with headers only; generate real content with:

```bash
python scripts/build_manifests.py --dataset-root data/processed --seed 0
```

## Expected dataset layout

Ultralytics-compatible (label discovery substitutes `images/` -> `labels/`):

```
data/processed/
├── images/
│   ├── mediterranean_scrubland/<session>/IMG_0001.jpg
│   ├── temperate_broadleaf/<session>/...
│   ├── boreal_conifer/<session>/...
│   └── custom_uav/<session>/...
└── labels/
    ├── mediterranean_scrubland/<session>/IMG_0001.txt   (YOLO format)
    └── ...
```

- `source` — first directory under `images/` (the biome/dataset row).
- `session` — the full sub-path below `source` (e.g. one flight). By default
  an entire session is assigned to a single split to avoid adjacent-frame
  leakage between train and test.
- Labels use YOLO format with class `0 = fire`, `1 = smoke`.

## dataset_manifest.csv

| Column | Meaning |
|---|---|
| `image_path` | Path relative to `images/` |
| `source` | Top-level source/biome directory |
| `session` | Containing sub-path (grouping key for the leakage guard) |
| `width`, `height` | Image dimensions (requires Pillow; blank otherwise) |
| `file_size_bytes`, `sha256` | Frozen file identity |
| `n_fire`, `n_smoke` | Instance counts parsed from the YOLO label file |
| `n_other` | Lines with class ids outside {0, 1} (flag for annotation errors) |

## split_manifest.csv

| Column | Meaning |
|---|---|
| `image_path`, `source`, `session` | As above |
| `split` | `train` / `val` / `test` (seeded 80/15/5 per source) |
| `n_fire`, `n_smoke` | Copied for oversampling and frame-level labeling |

## Rules

- Split is deterministic for a given seed and file set; re-running with the
  same inputs yields byte-identical CSVs.
- Images without a label file are counted in the build stats and labeled with
  zero instances — inspect them before training.
- Do not commit restricted imagery; commit only these manifests (paths +
  hashes), unless every upstream license permits redistribution.
- Record the manifest files' own hashes in any run manifest
  (`scripts/train_detector.py` does this for the split manifest).
# SPDX-License-Identifier: Apache-2.0
"""Ultralytics dataset assembly from split manifests.

Converts the repository's split manifest into the list files and
``dataset.yaml`` that Ultralytics expects, with 2x oversampling of fire-only
frames (fire present, smoke absent): each fire-only frame is appended once
more (ratio 2.0 -> 1 extra copy), i.e. it appears twice as often in the
training stream. Applied counts are written into the returned stats so they
land in the run manifest.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

YOLO_NAMES = {0: "fire", 1: "smoke"}


def read_split_rows(path) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("split manifest not found: {}".format(p))
    with open(p, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("split manifest is empty: {}".format(p))
    return rows


def _resolve_image_path(dataset_root: Path, image_path: str) -> str:
    p = Path(image_path)
    return str(p if p.is_absolute() else (dataset_root / image_path).resolve())


def build_ultralytics_dataset_yaml(
    split_manifest,
    out_dir,
    dataset_root=None,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    oversample_fire_only_ratio: float = 1.0,
) -> Dict[str, object]:
    """Write ``<split>.txt`` list files and a ``dataset.yaml``.

    Returns ``{"yaml_path", "list_paths", "stats"}``. Requires at least the
    ``train`` and ``val`` splits to be present for Ultralytics training.
    """
    import yaml

    rows = read_split_rows(split_manifest)
    dataset_root = Path(dataset_root) if dataset_root is not None else Path.cwd()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["split"]].append(row)

    list_paths: Dict[str, Path] = {}
    stats: Dict[str, object] = {}
    for split in splits:
        members = grouped.get(split, [])
        if not members:
            continue
        entries = [_resolve_image_path(dataset_root, r["image_path"]) for r in members]

        n_extra = 0
        if split == "train" and oversample_fire_only_ratio and oversample_fire_only_ratio > 1.0:
            extra_copies = int(round(oversample_fire_only_ratio)) - 1
            if extra_copies > 0:
                duplicated = []
                for row, entry in zip(members, entries):
                    if int(row.get("n_fire", 0) or 0) > 0 and int(row.get("n_smoke", 0) or 0) == 0:
                        duplicated.extend([entry] * extra_copies)
                entries.extend(duplicated)
                n_extra = len(duplicated)

        list_path = out_dir / "{}.txt".format(split)
        list_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        list_paths[split] = list_path
        stats[split] = {
            "rows": len(members),
            "listed_entries": len(entries),
            "fire_only_extra_copies": n_extra if split == "train" else 0,
        }

    if "train" not in list_paths or "val" not in list_paths:
        raise ValueError(
            "split manifest must contain train and val rows for Ultralytics "
            "(found: {})".format(sorted(list_paths))
        )

    dataset_yaml = {
        "path": str(dataset_root),
        "train": str(list_paths["train"]),
        "val": str(list_paths["val"]),
        "names": {str(k): v for k, v in YOLO_NAMES.items()},
    }
    if "test" in list_paths:
        dataset_yaml["test"] = str(list_paths["test"])

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    return {"yaml_path": str(yaml_path), "list_paths": {k: str(v) for k, v in list_paths.items()}, "stats": stats}

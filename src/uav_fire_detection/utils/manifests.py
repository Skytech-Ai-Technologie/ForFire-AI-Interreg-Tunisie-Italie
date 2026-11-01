# SPDX-License-Identifier: Apache-2.0
"""Versioned dataset and split manifest builders.

Produces deterministic, hash-audited manifests for the assembled corpus.

Expected layout (Ultralytics-compatible so label discovery works out of the
box)::

    dataset_root/images/<source>/<session...>/<image>.jpg
    dataset_root/labels/<source>/<session...>/<image>.txt   (YOLO format)

* ``source``  — first path component under ``images/`` (biome/dataset row).
* ``session`` — the full sub-path below ``source``; with
  ``group_by_session=True`` an entire session (e.g. one flight) lands in a
  single split, guarding against adjacent-frame leakage.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
YOLO_CLASS_NAMES = {0: "fire", 1: "smoke"}

DATASET_COLUMNS = [
    "image_path", "source", "session", "width", "height",
    "file_size_bytes", "sha256", "n_fire", "n_smoke", "n_other",
]
SPLIT_COLUMNS = [
    "image_path", "source", "session", "split", "n_fire", "n_smoke",
]

DEFAULT_SPLIT_FRACTIONS = (0.80, 0.15, 0.05)


def parse_yolo_label_file(path: Path) -> Tuple[int, int, int, int]:
    """Return (n_fire, n_smoke, n_other, n_invalid_lines) from a YOLO label file."""
    n_fire = n_smoke = n_other = n_invalid = 0
    if not path.exists():
        return n_fire, n_smoke, n_other, n_invalid
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            n_invalid += 1
            continue
        if class_id == 0:
            n_fire += 1
        elif class_id == 1:
            n_smoke += 1
        else:
            n_other += 1
    return n_fire, n_smoke, n_other, n_invalid


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dims(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height
    except Exception:  # noqa: BLE001 - dims are optional metadata
        return None, None


def iter_image_files(images_root: Path) -> List[Path]:
    files = [
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def build_manifests(
    dataset_root,
    labels_root=None,
    out_dir=None,
    seed: int = 0,
    group_by_session: bool = True,
    split_fractions: Tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
    images_dirname: str = "images",
    labels_dirname: str = "labels",
) -> Dict[str, object]:
    """Build dataset_manifest.csv + split_manifest.csv deterministically.

    Splits are allocated per (source, session) group with a seeded greedy
    assignment (largest groups first, to the split with the largest remaining
    deficit). The output is deterministic for a given seed and file set and
    is hashed into training run manifests.
    """
    dataset_root = Path(dataset_root)
    total = sum(split_fractions)
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split_fractions must sum to 1.0 (got {})".format(total))

    images_root = dataset_root / images_dirname
    if labels_root is None:
        labels_root = dataset_root / labels_dirname
    labels_root = Path(labels_root)
    if not images_root.is_dir():
        raise FileNotFoundError("images directory not found: {}".format(images_root))

    rows: List[Dict[str, object]] = []
    unlabeled = 0
    invalid_label_lines = 0
    for image_path in iter_image_files(images_root):
        rel = image_path.relative_to(images_root).as_posix()
        parts = rel.split("/")
        source = parts[0]
        session = "/".join(parts[:-1])
        label_path = labels_root / Path(rel).with_suffix(".txt")
        n_fire, n_smoke, n_other, n_invalid = parse_yolo_label_file(label_path)
        if not label_path.exists():
            unlabeled += 1
        invalid_label_lines += n_invalid
        width, height = _image_dims(image_path)
        rows.append(
            {
                "image_path": rel,
                "source": source,
                "session": session,
                "width": width,
                "height": height,
                "file_size_bytes": image_path.stat().st_size,
                "sha256": _sha256_file(image_path),
                "n_fire": n_fire,
                "n_smoke": n_smoke,
                "n_other": n_other,
            }
        )
    if not rows:
        raise ValueError("no images found under {}".format(images_root))

    out_dir = Path(out_dir) if out_dir is not None else dataset_root.parent / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = out_dir / "dataset_manifest.csv"
    _write_csv(dataset_csv, DATASET_COLUMNS, rows)

    split_rows = _assign_splits(rows, seed, group_by_session, split_fractions)
    split_csv = out_dir / "split_manifest.csv"
    _write_csv(split_csv, SPLIT_COLUMNS, split_rows)

    stats: Dict[str, object] = {
        "n_images": len(rows),
        "n_unlabeled_images": unlabeled,
        "n_invalid_label_lines": invalid_label_lines,
        "splits": {
            split: sum(1 for r in split_rows if r["split"] == split)
            for split in ("train", "val", "test")
        },
        "per_source_splits": {
            "{}/{}".format(r_source, r_split): sum(
                1 for r in split_rows
                if r["source"] == r_source and r["split"] == r_split
            )
            for r_source in sorted({r["source"] for r in split_rows})
            for r_split in ("train", "val", "test")
        },
        "dataset_manifest": str(dataset_csv),
        "split_manifest": str(split_csv),
    }
    return stats


def _assign_splits(
    rows: List[Dict[str, object]],
    seed: int,
    group_by_session: bool,
    split_fractions: Tuple[float, float, float],
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["source"]), str(row["session"]) if group_by_session else row["image_path"])
        groups[key].append(row)

    per_source: Dict[str, List[Tuple[str, List[Dict[str, object]]]]] = defaultdict(list)
    for (source, session), members in groups.items():
        per_source[source].append((session, members))

    split_names = ("train", "val", "test")
    split_rows: List[Dict[str, object]] = []
    for source in sorted(per_source):
        sessions = sorted(per_source[source], key=lambda kv: kv[0])
        rng = random.Random("{}|{}".format(seed, source))
        rng.shuffle(sessions)
        # Largest groups first: greedy assignment lands closest to fractions.
        sessions.sort(key=lambda kv: len(kv[1]), reverse=True)

        source_total = sum(len(members) for _, members in sessions)
        assigned = {"train": 0, "val": 0, "test": 0}
        targets = {name: frac * source_total for name, frac in zip(split_names, split_fractions)}

        for _, members in sessions:
            deficits = {name: targets[name] - assigned[name] for name in split_names}
            chosen = max(split_names, key=lambda name: (deficits[name], split_names.index(name)))
            assigned[chosen] += len(members)
            for row in members:
                split_rows.append(
                    {
                        "image_path": row["image_path"],
                        "source": row["source"],
                        "session": row["session"],
                        "split": chosen,
                        "n_fire": row["n_fire"],
                        "n_smoke": row["n_smoke"],
                    }
                )

    split_rows.sort(key=lambda r: str(r["image_path"]))
    return split_rows


def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_split_manifest(path) -> List[Dict[str, str]]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(
            "split manifest not found: {} — build one with scripts/build_manifests.py".format(split_path)
        )
    with open(split_path, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError("split manifest is empty: {}".format(split_path))
    return rows

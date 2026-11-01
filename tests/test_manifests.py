# SPDX-License-Identifier: Apache-2.0
"""Manifest builder and Ultralytics dataset assembly tests."""

import csv
from pathlib import Path

import pytest

from uav_fire_detection.utils.dataset import build_ultralytics_dataset_yaml
from uav_fire_detection.utils.manifests import build_manifests, parse_yolo_label_file

SPLIT_HEADER = ["image_path", "source", "session", "split", "n_fire", "n_smoke"]


def make_dataset_tree(root):
    images = root / "images"
    labels = root / "labels"
    for rel in ("srcA/s1/a.jpg", "srcA/s1/b.jpg", "srcA/s2/c.jpg", "srcB/s3/d.jpg"):
        path = images / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-jpeg-" + rel.encode())
    (labels / "srcA" / "s1").mkdir(parents=True)
    (labels / "srcA" / "s1" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.2 0.2\n")
    (labels / "srcA" / "s1" / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (labels / "srcA" / "s2").mkdir(parents=True)
    (labels / "srcA" / "s2" / "c.txt").write_text("")
    # srcB/s3/d.txt intentionally missing -> unlabeled image


def test_parse_yolo_label_counts(tmp_path):
    path = tmp_path / "l.txt"
    path.write_text("0 0.5 0.5 0.1 0.1\n1 0.5 0.5 0.2 0.2\n\n5 0.1 0.1 0.1 0.1\nbad\n")
    assert parse_yolo_label_file(path) == (1, 1, 1, 1)

    missing = tmp_path / "missing.txt"
    assert parse_yolo_label_file(missing) == (0, 0, 0, 0)


def test_build_manifests_is_deterministic_and_grouped(tmp_path):
    make_dataset_tree(tmp_path)
    out1, out2 = tmp_path / "m1", tmp_path / "m2"
    stats1 = build_manifests(tmp_path, out_dir=out1, seed=0)
    stats2 = build_manifests(tmp_path, out_dir=out2, seed=0)

    assert stats1["n_images"] == 4
    assert stats1["n_unlabeled_images"] == 1
    assert (out1 / "split_manifest.csv").read_text(encoding="utf-8") == \
           (out2 / "split_manifest.csv").read_text(encoding="utf-8")
    assert sum(stats1["splits"].values()) == 4

    with open(out1 / "split_manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert all(r["split"] in ("train", "val", "test") for r in rows)
    # Whole sessions land in one split (leakage guard).
    per_session = {}
    for r in rows:
        per_session.setdefault((r["source"], r["session"]), set()).add(r["split"])
    assert all(len(splits) == 1 for splits in per_session.values())

    with open(out1 / "dataset_manifest.csv", newline="", encoding="utf-8") as fh:
        dataset_rows = {r["image_path"]: r for r in csv.DictReader(fh)}
    assert dataset_rows["srcA/s1/a.jpg"]["n_fire"] == "1"
    assert dataset_rows["srcA/s1/a.jpg"]["n_smoke"] == "1"
    assert dataset_rows["srcA/s1/b.jpg"]["sha256"]
    assert dataset_rows["srcB/s3/d.jpg"]["n_fire"] == "0"


def test_build_manifests_rejects_fraction_mismatch(tmp_path):
    with pytest.raises(ValueError):
        build_manifests(tmp_path, out_dir=tmp_path / "m", split_fractions=(0.5, 0.2, 0.2))


def test_dataset_yaml_fire_only_oversampling(tmp_path):
    manifest = tmp_path / "split_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SPLIT_HEADER)
        writer.writeheader()
        writer.writerow({"image_path": "images/a.jpg", "source": "s", "session": "x",
                         "split": "train", "n_fire": "1", "n_smoke": "0"})
        writer.writerow({"image_path": "images/b.jpg", "source": "s", "session": "x",
                         "split": "train", "n_fire": "2", "n_smoke": "0"})
        writer.writerow({"image_path": "images/c.jpg", "source": "s", "session": "x",
                         "split": "train", "n_fire": "1", "n_smoke": "1"})
        writer.writerow({"image_path": "images/d.jpg", "source": "s", "session": "x",
                         "split": "train", "n_fire": "0", "n_smoke": "0"})
        writer.writerow({"image_path": "images/e.jpg", "source": "s", "session": "x",
                         "split": "val", "n_fire": "0", "n_smoke": "0"})
        writer.writerow({"image_path": "images/f.jpg", "source": "s", "session": "x",
                         "split": "test", "n_fire": "1", "n_smoke": "0"})

    result = build_ultralytics_dataset_yaml(
        manifest, out_dir=tmp_path / "ds", dataset_root=tmp_path,
        oversample_fire_only_ratio=2.0,
    )

    train_entries = (tmp_path / "ds" / "train.txt").read_text(encoding="utf-8").splitlines()
    # 4 unique train rows + 1 extra copy for each of the two fire-only frames.
    assert len(train_entries) == 6
    assert train_entries.count(str((tmp_path / "images" / "a.jpg").resolve())) == 2
    assert train_entries.count(str((tmp_path / "images" / "b.jpg").resolve())) == 2
    assert train_entries.count(str((tmp_path / "images" / "c.jpg").resolve())) == 1

    import yaml

    dataset_yaml = yaml.safe_load((tmp_path / "ds" / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["names"] == {"0": "fire", "1": "smoke"}
    assert dataset_yaml["train"].endswith("train.txt")
    assert dataset_yaml["val"].endswith("val.txt")
    assert dataset_yaml["test"].endswith("test.txt")
    assert result["stats"]["train"]["fire_only_extra_copies"] == 2

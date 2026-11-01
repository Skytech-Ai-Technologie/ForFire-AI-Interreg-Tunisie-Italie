# SPDX-License-Identifier: Apache-2.0
"""Environment capture.

Records the exact software/hardware environment (Python, JetPack/L4T, CUDA,
package versions, container detection, git commit) into a JSON snapshot so
every reported result can name its environment.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except Exception:  # noqa: BLE001 - absence is a valid snapshot result
        return None


def _jetpack_info() -> Dict[str, Any]:
    """JetPack/L4T release from /etc/nv_tegra_release (Jetson only)."""
    info: Dict[str, Any] = {"detected": False}
    path = Path("/etc/nv_tegra_release")
    if path.exists():
        try:
            first = path.read_text(encoding="utf-8").splitlines()[0].strip()
            info = {"detected": True, "release_line": first}
        except OSError:  # pragma: no cover - unreadable file
            pass
    return info


def _git_commit(repo_dir) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a repo / git missing
        return None


def collect(repo_dir=None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jetpack": _jetpack_info(),
        "in_docker": Path("/.dockerenv").exists(),
    }
    if repo_dir is not None:
        info["git_commit"] = _git_commit(repo_dir)

    packages = {}
    for dist in ("torch", "torchvision", "ultralytics", "open_clip_torch", "opencv-python", "numpy", "pymavlink"):
        packages[dist] = _package_version(dist)
    info["packages"] = packages

    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        info["cuda_available"] = None

    return info


def write_snapshot(path, repo_dir=None) -> Dict[str, Any]:
    info = collect(repo_dir=repo_dir)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
    return info

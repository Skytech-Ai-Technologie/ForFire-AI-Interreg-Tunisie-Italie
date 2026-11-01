# SPDX-License-Identifier: Apache-2.0
"""Seed management.

Seeds python/numpy/torch where available and records what was actually
applied, so run manifests can state the RNG posture.
"""

from __future__ import annotations

import random
from typing import Any, Dict


def set_seed(seed: int, deterministic: bool = True) -> Dict[str, Any]:
    """Seed python/numpy/torch where available; return what was configured."""
    applied: Dict[str, Any] = {"seed": int(seed), "deterministic": bool(deterministic)}

    random.seed(seed)
    applied["python"] = True

    try:
        import numpy as np

        np.random.seed(seed)
        applied["numpy"] = np.__version__
    except ImportError:  # pragma: no cover - optional at runtime
        applied["numpy"] = None

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not deterministic
        applied["torch"] = torch.__version__
    except ImportError:  # pragma: no cover - optional at runtime
        applied["torch"] = None

    return applied

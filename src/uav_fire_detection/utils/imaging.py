# SPDX-License-Identifier: Apache-2.0
"""Image I/O helpers.

``imread_bgr`` goes through ``np.fromfile`` + ``cv2.imdecode`` so paths with
non-ASCII characters work on Windows (``cv2.imread`` fails silently there).
"""

from __future__ import annotations


def imread_bgr(path):
    """Read an image as a HxWx3 BGR array; raises on unreadable input."""
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError("unreadable image: {}".format(path))
    return image

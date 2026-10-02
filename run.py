#!/usr/bin/env python3
"""
Deep Face Net - Entry Point
Launches GUI by default, or runs CLI mode when arguments are provided
"""

import os

# Media Foundation's hardware transforms make the camera take ~2x longer to deliver its first
# frame (measured 1.4-1.8 s -> 0.6-0.85 s). Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

from core.main import main  # noqa: E402

if __name__ == "__main__":
    main()

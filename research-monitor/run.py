#!/usr/bin/env python3
"""Entry point for Research Monitor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.monitor import main

if __name__ == "__main__":
    main()

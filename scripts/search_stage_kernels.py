#!/usr/bin/env python3
"""Independent CLI; intentionally never imports main.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernel_stage_search.cli import main

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the actual C and training encoder inputs in their owning environments."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
subprocess.run([
    str(ROOT / 'training/posttrain/.venv/bin/python'),
    str(ROOT / 'training/posttrain/tests/test_encoder_input.py'),
    sys.argv[1],
], check=True)

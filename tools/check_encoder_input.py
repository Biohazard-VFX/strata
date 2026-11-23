#!/usr/bin/env python3
"""Compare the actual C and training encoder inputs in their owning environments."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory() as temporary:
    oracle = str(Path(temporary) / 'encoder.npz')
    subprocess.run([
        str(ROOT / 'training/posttrain/.venv/bin/python'),
        str(ROOT / 'training/posttrain/tests/test_encoder_input.py'),
        sys.argv[1], oracle,
    ], check=True)
    subprocess.run([
        str(ROOT / 'training/.venv/bin/python'),
        str(ROOT / 'training/tests/test_reference.py'), oracle,
    ], check=True)

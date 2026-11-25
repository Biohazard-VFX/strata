#!/usr/bin/env python3
"""Compare the actual C and training encoder inputs in their owning environments."""
import json
from pathlib import Path

from strata_contract import spec
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
matrix = json.loads(subprocess.check_output([
    str(ROOT / 'training/posttrain/.venv/bin/python'), '-c',
    'import json; from strata_training.data.encoder_input import DISPLAY_MATRIX; print(json.dumps(DISPLAY_MATRIX))',
], text=True))
assert matrix == spec()['preprocessing']['training_display_matrix'], 'Training display matrix differs from contract'
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

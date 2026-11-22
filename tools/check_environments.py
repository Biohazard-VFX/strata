#!/usr/bin/env python3
"""Prove each synced project environment can neither find nor hold a Strata package its layer forbids."""
import json
import os
from pathlib import Path
import subprocess
import sys

from boundaries import LAYERS, PROJECTS, distribution, is_strata

ROOT = Path(__file__).resolve().parent.parent
PROBE = '''
import importlib.metadata, importlib.util, json, sys
print(json.dumps({
    "modules": [m for m in sys.argv[1:] if importlib.util.find_spec(m) is not None],
    "distributions": sorted({d.metadata["Name"] for d in importlib.metadata.distributions()}),
}))
'''


def violations(directory, python):
    name, allowed = PROJECTS[directory]
    forbidden = next(roots for layer, roots, _ in LAYERS if layer == directory)
    env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'VIRTUAL_ENV')}
    result = subprocess.run([str(python), '-I', '-c', PROBE, *forbidden], capture_output=True, text=True,
                            check=True, cwd=ROOT, env=env)
    found = json.loads(result.stdout)
    errors = [f'{directory} environment ({python}) can import {module}' for module in found['modules']]
    errors += [f'{directory} environment ({python}) holds {d}' for d in map(distribution, found['distributions'])
               if is_strata(d) and d not in {name, *allowed}]
    return errors


def main():
    errors = violations('training', Path(sys.executable)) + violations('datagen', ROOT/'datagen'/'.venv'/'bin'/'python')
    if errors:
        sys.exit('\n'.join(errors))
    print('training and datagen environments hold and find only the Strata packages their layers allow')


if __name__ == '__main__':
    main()

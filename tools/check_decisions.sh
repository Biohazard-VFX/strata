#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import subprocess
import sys

errors = []
files = [Path(name) for name in subprocess.check_output(['git', 'ls-files', '--cached', '--others',
                                                          '--exclude-standard'], text=True).split()
         if Path(name).name == 'DECISIONS.md']
if not files:
    sys.exit('no DECISIONS.md found')
for path in files:
    lines = path.read_text().splitlines()
    if not lines or lines[0] != '# Decisions':
        errors.append(f'{path}:1: must start with "# Decisions"')
    for number, line in enumerate(lines[1:], 2):
        if not line:
            continue
        if not line.startswith('- ') or line.startswith('-  '):
            errors.append(f'{path}:{number}: entries are single "- " lines')
        if len(line) > 202:
            errors.append(f'{path}:{number}: entry exceeds 200 characters')
if errors:
    sys.exit('\n'.join(errors))
print(f'decision entries passed in {len(files)} file(s)')
PY

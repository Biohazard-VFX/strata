#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, 'tools')
from source_scan import c_comments, python_comments

marker = re.compile(r'\b(?:TODO|FIXME|XXX|HACK|workaround)\b', re.I)
listed = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
                                  'runtime', 'training', 'tools', 'contract'], text=True).split()
errors = []
checked = 0
for name in sorted(set(listed)):
    path = Path(name)
    if path.suffix not in ('.c', '.h', '.py') or not path.is_file():
        continue
    if path.is_relative_to('runtime/third_party'):
        continue
    checked += 1
    text = path.read_text()
    if marker.search(text):
        errors.append(f'{path}: forbidden marker')
    if path.suffix == '.py':
        comments = [(line, line, body) for line, body in python_comments(text)]
    else:
        comments = c_comments(text)
    previous = -1
    for first, last, _ in comments:
        if last > first:
            errors.append(f'{path}:{first}: comment exceeds one line')
        if first == previous + 1:
            errors.append(f'{path}:{first}: consecutive comment lines')
        previous = last
if not checked:
    sys.exit('no sources found')
if errors:
    sys.exit('\n'.join(errors))
print(f'comment checks passed on {checked} C/Python files')
PY

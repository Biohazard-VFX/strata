#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import re
import sys

errors = []
for path in Path('runtime/src').rglob('*'):
    if path.suffix not in ('.c','.h'):
        continue
    text = path.read_text()
    if re.search(r'\b(?:TODO|FIXME|XXX|HACK|workaround)\b', text, re.I):
        errors.append(f'{path}: forbidden marker')
    for comment in re.finditer(r'/\*.*?\*/', text, re.S):
        if comment[0].count('\n')+1 > 2:
            errors.append(f'{path}:{text[:comment.start()].count(chr(10))+1}: comment exceeds two lines')
if errors:
    sys.exit('\n'.join(errors))
print('comment checks passed')
PY

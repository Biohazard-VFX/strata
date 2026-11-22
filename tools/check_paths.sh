#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import re
import subprocess
import sys

anchor = r'(?<![\w.~$}/-])/'
patterns = (
    (re.compile(anchor + r'(?:usr/(?:local/)?)?s?bin/(?!env\b)[\w.+-]+'), 'absolute interpreter path'),
    (re.compile(anchor + r'(?:home|mnt|media|Users|root|opt)/'), 'machine-specific absolute path'),
)
errors = []
checked = 0
for name in sorted(set(subprocess.check_output(['git', 'ls-files', '--cached', '--others',
                                                '--exclude-standard'], text=True).split())):
    path = Path(name)
    if not path.is_file():
        continue
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    checked += 1
    for number, line in enumerate(text.splitlines(), 1):
        for pattern, message in patterns:
            match = pattern.search(line)
            if match:
                errors.append(f'{path}:{number}: {message}: {match[0]}')
if not checked:
    sys.exit('no text files found')
if errors:
    sys.exit('\n'.join(errors))
print(f'no machine-specific paths in {checked} files')
PY

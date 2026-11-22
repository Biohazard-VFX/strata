#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import os
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
    if path.is_symlink():
        target = os.readlink(path)
        if os.path.isabs(target):
            errors.append(f'{path}: symlink target is absolute: {target}')
        elif os.path.normpath(os.path.join(path.parent, target)).split(os.sep)[0] == '..':
            errors.append(f'{path}: symlink target leaves the repository: {target}')
        continue
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
print(f'no machine-specific paths in {checked} files and no symlink leaves the repository')
PY

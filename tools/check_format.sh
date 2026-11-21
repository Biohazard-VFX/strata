#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from pathlib import Path
import subprocess
files = sorted(str(p) for d in ('runtime/src','runtime/include','runtime/tests')
               for p in Path(d).rglob('*') if p.suffix in ('.c','.h'))
assert files, 'no C files found'
subprocess.run(['clang-format','--dry-run','--Werror',*files],check=True)
PY

#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ $# -eq 0 ]]; then
  exec python3 "$root/tools/check_commits.py" "${STRATA_COMMIT_RANGE:-origin/main..HEAD}"
fi
exec python3 "$root/tools/check_commits.py" "$@"

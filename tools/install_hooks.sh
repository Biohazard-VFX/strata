#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
expected=$(cd "$(dirname "$0")/.." && pwd)
[[ "$root" == "$expected" ]] || { printf 'run this from the Strata checkout\n' >&2; exit 1; }
old=$(git config --local --get core.hooksPath || true)
hooks="$root/tools/hooks"
[[ -z "$old" || "$old" == "$hooks" ]] || { printf 'refusing to replace existing hooksPath: %s\n' "$old" >&2; exit 1; }
[[ -x "$hooks/commit-msg" ]] || { printf 'commit-msg hook is missing or not executable\n' >&2; exit 1; }
git config --local core.hooksPath "$hooks"
[[ $(git config --local --get core.hooksPath) == "$hooks" ]]
printf 'installed commit-msg hook: %s/commit-msg\n' "$hooks"

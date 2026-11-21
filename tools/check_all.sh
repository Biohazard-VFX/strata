#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
python=${STRATA_TEST_PYTHON:-"$root/training/.venv/bin/python"}
export PYTHONPATH=
bash tools/check_commits.sh
"$python" tools/check_commit_rules.py
"$python" tools/check_rule_cases.py
bash tools/check_layers.sh
bash tools/check_comments.sh
bash tools/check_decisions.sh
bash tools/check_format.sh
"$python" -m ruff check training contract --config training/pyproject.toml
"$python" -m ruff format --check training contract --config training/pyproject.toml
"$python" -m pyright --project training
"$python" -m unittest discover -s training/tests -v
"$python" tools/check_contract.py
presets=(normal sanitize)
cpu=0
if [[ $# -gt 0 ]]; then
  case "$1" in normal|sanitize) presets=("$1");; --cpu) presets=(cpu); cpu=1;; *) printf 'usage: %s [normal|sanitize|--cpu]\n' "$0" >&2; exit 2;; esac
fi
for preset in "${presets[@]}"; do
  options=("-DSTRATA_TEST_PYTHON=$python")
  if [[ -n ${ONNXRUNTIME_ROOT:-} ]]; then options+=("-DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT"); fi
  if [[ -n ${STRATA_MODEL:-} ]]; then options+=("-DSTRATA_MODEL=$STRATA_MODEL"); fi
  cmake --preset "$preset" -S runtime "${options[@]}"
  "$python" - "$root/build/$preset/CMakeCache.txt" "$preset" <<'PY'
import sys
from pathlib import Path
cache = Path(sys.argv[1]).read_text()
expected = "ON" if sys.argv[2] == "sanitize" else "OFF"
assert f"STRATA_SANITIZE:BOOL={expected}\n" in cache, "CMake reset options; reconfigure before qualification"
PY
  cmake --build "$root/build/$preset" --parallel 4
  tests=()
  if [[ "$cpu" == 1 ]]; then tests=(-LE gpu); fi
  if [[ "$preset" == sanitize ]]; then
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
      ctest --test-dir "$root/build/$preset" --output-on-failure "${tests[@]}"
  else
    ctest --test-dir "$root/build/$preset" --output-on-failure "${tests[@]}"
  fi
done

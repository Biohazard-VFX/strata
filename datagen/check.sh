#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python=.venv/bin/python
export PYTHONPATH=
"$python" -m ruff format --check . ../contract/dataset
"$python" -m ruff check . ../contract/dataset
"$python" -m pyright
"$python" tools/check_generators.py strata_datagen/generators
"$python" tools/check_identifiers.py . ../contract/dataset
"$python" tools/check_decisions.py DECISIONS.md
"$python" -m unittest discover -s tests

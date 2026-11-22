#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
mkdir -p build
fixture=$(mktemp -d "$root/build/training-fixture.XXXXXX")
trap 'rm -rf "$fixture"' EXIT
datagen/.venv/bin/python datagen/tests/training_fixture.py "$fixture/shot"
STRATA_DATASET_FIXTURE="$fixture/shot" bash training/posttrain/check.sh

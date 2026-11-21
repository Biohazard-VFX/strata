#!/usr/bin/env python3
"""Gate native CUDA network output against PyTorch oracles; never rescale predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import OpenEXR
from PIL import Image

from deep_readback import verify_deep

SKIP = 77


def errors(native, reference):
    assert native.shape == reference.shape, (native.shape, reference.shape)
    assert np.isfinite(native).all() and (native >= 0).all()
    np.testing.assert_array_equal(native == 0, reference == 0)
    positive = reference > 0
    assert positive.any()
    return (native[positive] - reference[positive]) / reference[positive]


def suppressions(stderr):
    match = re.search(r'Suppressions used:\n\s*count\s+bytes\s+template\n(.*?)\n-{10,}', stderr, re.S)
    return [] if not match else [line.split()[-1] for line in match[1].splitlines() if line.strip()]


def run_case(args, case, data, prefix):
    command = [str(args.cli), 'depth', '--model', str(args.model), '--in', str(data/case['input']),
               '--out-depth', str(prefix.with_suffix('.depth.exr')),
               '--out-deep', str(prefix.with_suffix('.deep.exr')),
               '--provider', 'cuda', '--exposure', str(case.get('exposure', 0))]
    if not case.get('auto_focal', False):
        command += ['--f-px', str(case['focal_px'])]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    prefix.with_suffix('.stderr.log').write_text(completed.stderr)
    assert completed.returncode == 0, completed.stderr
    if args.lsan_suppression:
        used = suppressions(completed.stderr)
        assert used == [args.lsan_suppression], ('unexpected LSan suppressions', used)
    report = json.loads(completed.stdout)
    assert report['provider'] == 'cuda' and not report['fallback_reason'], report
    return report


def check_case(args, case, data, reference_dir):
    prefix = args.out/case['id']
    plate = data/case['input']
    with plate.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    with np.load(reference_dir/(case['id']+'.npz')) as archive:
        reference = {key: archive[key] for key in archive.files}
    assert str(reference['input_sha256']) == digest, 'stale PyTorch oracle'
    assert float(reference['exposure']) == case.get('exposure', 0)
    assert bool(reference['auto_focal']) == case.get('auto_focal', False)
    report = run_case(args, case, data, prefix)
    focal = float(reference['focal_px'])
    assert abs(report['focal_px']-focal)/focal <= args.focal_relative, (report['focal_px'], focal)
    with OpenEXR.File(str(prefix.with_suffix('.depth.exr')), separate_channels=True) as file:
        assert set(file.channels()) == {'Z', 'network_inverse_depth'}
        native = file.channels()['network_inverse_depth'].pixels.copy()
        z = file.channels()['Z'].pixels.copy()
    error = errors(native, reference['network_inverse_depth'])
    stats = {'max_relative': float(np.abs(error).max()), 'median_relative': float(np.median(np.abs(error))),
             'p99_relative': float(np.quantile(np.abs(error), .99)), 'signed_bias': float(error.mean()),
             'positive_fraction': float((error > 0).mean()), 'pixels': int(error.size)}
    return {'id': case['id'], 'timing': report, 'network_inverse_depth': stats}, error, z


def gate(row, args):
    stats = row['network_inverse_depth']
    failures = []
    if stats['max_relative'] > args.max_relative:
        failures.append('max')
    if stats['median_relative'] > args.median_relative:
        failures.append('median')
    if abs(stats['signed_bias']) > args.signed_bias:
        failures.append('signed bias')
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lsan-suppression')
    parser.add_argument('--max-relative', type=float, default=.05)
    parser.add_argument('--median-relative', type=float, default=.001)
    parser.add_argument('--signed-bias', type=float, default=.001)
    parser.add_argument('--focal-relative', type=float, default=.001)
    args = parser.parse_args()
    if not os.environ.get('STRATA_PARITY_DATA') or not os.environ.get('STRATA_PARITY_REFERENCE'):
        print('skipped: set STRATA_PARITY_DATA and STRATA_PARITY_REFERENCE to run CUDA parity')
        sys.exit(SKIP)
    data = Path(os.environ['STRATA_PARITY_DATA'])
    reference_dir = Path(os.environ['STRATA_PARITY_REFERENCE'])
    cases = json.loads(args.cases.read_text())
    assert cases and len({case['id'] for case in cases}) == len(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, failed, all_errors = [], [], []
    for index, case in enumerate(cases):
        row, error, z = check_case(args, case, data, reference_dir)
        if index == 0:
            rgb = np.asarray(Image.open(data/case['input']).convert('RGB'), dtype=np.float32)/255
            verify_deep(args.out/(case['id']+'.deep.exr'), rgb, z)
        row['failed'] = gate(row, args)
        failed += [(case['id'], row['failed'])] if row['failed'] else []
        rows.append(row)
        all_errors.append(np.abs(error))
        (args.out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps({'id': row['id'], **row['network_inverse_depth'], 'failed': row['failed']}), flush=True)
    pooled = np.concatenate(all_errors)
    summary = {'cases': len(rows), 'failed': failed,
               'aggregate': {'max_relative': float(pooled.max()), 'median_relative': float(np.median(pooled)),
                             'p99_relative': float(np.quantile(pooled, .99)), 'pixels': int(pooled.size)},
               'worst_signed_bias': max((row['network_inverse_depth']['signed_bias'] for row in rows), key=abs)}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    assert not failed, failed


if __name__ == '__main__':
    main()

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


def discontinuities(reference, threshold, radius):
    positive = reference > 0
    log = np.log(np.where(positive, reference, 1))
    edge = np.zeros(reference.shape, bool)
    for axis in (0, 1):
        step = (np.abs(np.diff(log, axis=axis)) > threshold) | (np.diff(positive.astype(np.int8), axis=axis) != 0)
        head = [slice(None), slice(None)]
        tail = [slice(None), slice(None)]
        head[axis], tail[axis] = slice(1, None), slice(None, -1)
        edge[tuple(head)] |= step
        edge[tuple(tail)] |= step
    padded = np.pad(edge, radius)
    near = np.zeros_like(edge)
    height, width = edge.shape
    for dy in range(2*radius + 1):
        for dx in range(2*radius + 1):
            near |= padded[dy:dy + height, dx:dx + width]
    return near


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
    oracle = reference['network_inverse_depth']
    error = errors(native, oracle)
    interior = ~discontinuities(oracle, args.edge_log_step, args.edge_radius)[oracle > 0]
    assert interior.any()
    stats = {'max_relative': float(np.abs(error).max()), 'interior_max_relative': float(np.abs(error[interior]).max()),
             'p999_relative': float(np.quantile(np.abs(error), .999)),
             'median_relative': float(np.median(np.abs(error))),
             'p99_relative': float(np.quantile(np.abs(error), .99)), 'signed_bias': float(error.mean()),
             'positive_fraction': float((error > 0).mean()), 'edge_fraction': float(1 - interior.mean()),
             'pixels': int(error.size)}
    return {'id': case['id'], 'timing': report, 'network_inverse_depth': stats}, error, z


def gate(row, case, args):
    stats = row['network_inverse_depth']
    limits = {'interior_max': (stats['interior_max_relative'], args.interior_max_relative),
              'p999': (stats['p999_relative'], args.p999_relative),
              'median': (stats['median_relative'], args.median_relative),
              'signed_bias': (abs(stats['signed_bias']), args.signed_bias)}
    quarantined = case.get('quarantine', {})
    assert set(quarantined) <= set(limits), quarantined
    return [name for name, (value, limit) in limits.items() if name not in quarantined and value > limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lsan-suppression')
    parser.add_argument('--interior-max-relative', type=float, default=.05)
    parser.add_argument('--p999-relative', type=float, default=.01)
    parser.add_argument('--edge-log-step', type=float, default=.3)
    parser.add_argument('--edge-radius', type=int, default=2)
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
        row['failed'] = gate(row, case, args)
        row['quarantine'] = sorted(case.get('quarantine', {}))
        failed += [(case['id'], row['failed'])] if row['failed'] else []
        rows.append(row)
        all_errors.append(np.abs(error))
        (args.out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps({'id': row['id'], **row['network_inverse_depth'], 'failed': row['failed'],
                          'quarantined': bool(row['quarantine'])}), flush=True)
    pooled = np.concatenate(all_errors)
    summary = {'cases': len(rows), 'failed': failed,
               'quarantined': [row['id'] for row in rows if row['quarantine']],
               'aggregate': {'max_relative': float(pooled.max()), 'median_relative': float(np.median(pooled)),
                             'p99_relative': float(np.quantile(pooled, .99)),
                             'p999_relative': float(np.quantile(pooled, .999)), 'pixels': int(pooled.size)},
               'worst_interior_max_relative': max(row['network_inverse_depth']['interior_max_relative'] for row in rows),
               'worst_p999_relative': max(row['network_inverse_depth']['p999_relative'] for row in rows),
               'worst_gated_signed_bias': max((row['network_inverse_depth']['signed_bias'] for row in rows
                                               if 'signed_bias' not in row['quarantine']), key=abs)}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    assert not failed, failed


if __name__ == '__main__':
    main()

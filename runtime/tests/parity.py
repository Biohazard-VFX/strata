#!/usr/bin/env python3
"""Gate native CUDA network output against PyTorch FP16 and FP32 oracles; never rescale predictions."""
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
EDGE_LOG_STEP = .3
EDGE_RADIUS = 2
FOCAL_RELATIVE = .001
LIMITS = {'interior_max': .05, 'p999': .01, 'median': .0025, 'signed_bias': .0025, 'fp32_margin': .002}
CASE_KEYS = {'id', 'input', 'focal_px', 'auto_focal', 'exposure'}


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


def statistics(native, reference):
    error = errors(native, reference)
    interior = ~discontinuities(reference, EDGE_LOG_STEP, EDGE_RADIUS)[reference > 0]
    assert interior.any()
    magnitude = np.abs(error)
    return {'max_relative': float(magnitude.max()), 'interior_max_relative': float(magnitude[interior].max()),
            'p999_relative': float(np.quantile(magnitude, .999)), 'median_relative': float(np.median(magnitude)),
            'p99_relative': float(np.quantile(magnitude, .99)), 'signed_bias': float(error.mean()),
            'positive_fraction': float((error > 0).mean()), 'edge_fraction': float(1 - interior.mean()),
            'pixels': int(error.size)}, magnitude


def load_reference(directory, case, digest, precision):
    with np.load(directory/(case['id']+'.npz')) as archive:
        reference = {key: archive[key] for key in archive.files}
    assert str(reference['input_sha256']) == digest, 'stale PyTorch oracle'
    assert str(reference['precision']) == precision, (case['id'], str(reference['precision']), precision)
    assert float(reference['exposure']) == case.get('exposure', 0)
    assert bool(reference['auto_focal']) == case.get('auto_focal', False)
    return reference


def check_case(args, case, data, references):
    prefix = args.out/case['id']
    with (data/case['input']).open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    reference = load_reference(references['fp16'], case, digest, 'fp16')
    truth = load_reference(references['fp32'], case, digest, 'fp32')['network_inverse_depth']
    report = run_case(args, case, data, prefix)
    focal = float(reference['focal_px'])
    assert abs(report['focal_px']-focal)/focal <= FOCAL_RELATIVE, (report['focal_px'], focal)
    with OpenEXR.File(str(prefix.with_suffix('.depth.exr')), separate_channels=True) as file:
        assert set(file.channels()) == {'Z', 'network_inverse_depth'}
        native = file.channels()['network_inverse_depth'].pixels.copy()
        z = file.channels()['Z'].pixels.copy()
    oracle = reference['network_inverse_depth']
    matching, magnitude = statistics(native, oracle)
    native_fp32, _ = statistics(native, truth)
    torch_fp32, _ = statistics(oracle, truth)
    return {'id': case['id'], 'timing': report, 'network_inverse_depth': matching,
            'fp32': {'native': native_fp32, 'torch_fp16': torch_fp32}}, magnitude, z


def gate(matching, native_fp32, torch_fp32):
    margin = LIMITS['fp32_margin']
    measured = {'interior_max': (matching['interior_max_relative'], LIMITS['interior_max']),
                'p999': (matching['p999_relative'], LIMITS['p999']),
                'median': (matching['median_relative'], LIMITS['median']),
                'signed_bias': (abs(matching['signed_bias']), LIMITS['signed_bias']),
                'fp32_median': (native_fp32['median_relative'], torch_fp32['median_relative'] + margin),
                'fp32_signed_bias': (abs(native_fp32['signed_bias']), abs(torch_fp32['signed_bias']) + margin)}
    return [name for name, (value, limit) in measured.items() if not value <= limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lsan-suppression')
    args = parser.parse_args()
    variables = {'data': 'STRATA_PARITY_DATA', 'fp16': 'STRATA_PARITY_REFERENCE', 'fp32': 'STRATA_PARITY_REFERENCE_FP32'}
    present = {key: os.environ.get(name) for key, name in variables.items()}
    if not any(present.values()):
        print('skipped: set ' + ', '.join(variables.values()) + ' to run CUDA parity')
        sys.exit(SKIP)
    missing = [variables[key] for key, value in present.items() if not value]
    assert not missing, f'parity needs every oracle: {missing} unset'
    data = Path(present['data'])
    references = {key: Path(present[key]) for key in ('fp16', 'fp32')}
    cases = json.loads(args.cases.read_text())
    assert cases and len({case['id'] for case in cases}) == len(cases)
    for case in cases:
        assert set(case) <= CASE_KEYS, (case['id'], set(case) - CASE_KEYS)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, failed, all_errors = [], [], []
    for index, case in enumerate(cases):
        row, magnitude, z = check_case(args, case, data, references)
        if index == 0:
            rgb = np.asarray(Image.open(data/case['input']).convert('RGB'), dtype=np.float32)/255
            verify_deep(args.out/(case['id']+'.deep.exr'), rgb, z)
        row['failed'] = gate(row['network_inverse_depth'], row['fp32']['native'], row['fp32']['torch_fp16'])
        failed += [(case['id'], row['failed'])] if row['failed'] else []
        rows.append(row)
        all_errors.append(magnitude)
        (args.out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps({'id': row['id'], **row['network_inverse_depth'], 'fp32': row['fp32'], 'failed': row['failed']}),
              flush=True)
    pooled = np.concatenate(all_errors)
    summary = {'cases': len(rows), 'failed': failed, 'limits': LIMITS,
               'aggregate': {'max_relative': float(pooled.max()), 'median_relative': float(np.median(pooled)),
                             'p99_relative': float(np.quantile(pooled, .99)),
                             'p999_relative': float(np.quantile(pooled, .999)), 'pixels': int(pooled.size)},
               'worst_interior_max_relative': max(row['network_inverse_depth']['interior_max_relative'] for row in rows),
               'worst_p999_relative': max(row['network_inverse_depth']['p999_relative'] for row in rows),
               'worst_signed_bias': max((row['network_inverse_depth']['signed_bias'] for row in rows), key=abs)}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    assert not failed, failed


if __name__ == '__main__':
    main()

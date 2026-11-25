import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import shutil
import OpenEXR
from strata_contract import write_manifest

ROOT = Path(__file__).resolve().parent.parent


def compare(path, reference, tolerance, colorspace):
    plate, inverse, alpha = (reference[name] for name in ('plate', 'inverse', 'alpha'))
    height, width = plate.shape[-2:]
    with OpenEXR.File(str(path), separate_channels=True) as file:
        assert file.header()['strata:color_space'] == colorspace
        expected_xy = {'acescg': (.713, .293, .165, .830, .128, .044, .32168, .33767),
            'aces2065-1': (.7347, .2653, 0, 1, .0001, -.077, .32168, .33767),
            'lin-rec709': (.64, .33, .30, .60, .15, .06, .3127, .3290),
            'lin-rec2020': (.708, .292, .170, .797, .131, .046, .3127, .3290)}
        np.testing.assert_allclose(file.header()['chromaticities'], expected_xy[colorspace], rtol=0, atol=1e-7)
        channels = {name: channel.pixels for name, channel in file.channels().items()}
        assert set(channels) == {'R', 'G', 'B', 'A', 'Z', 'ZBack'}
        values = {name: np.stack([channels[name][y, x] for y in range(height)
                  for x in range(width)]).reshape(height, width, 4) for name in channels}
    expected_z = (1 / inverse).transpose(1, 2, 0)
    expected_alpha = alpha.transpose(1, 2, 0)
    np.testing.assert_allclose(values['Z'], expected_z, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(values['Z'], values['ZBack'])
    assert np.isfinite(values['Z']).all() and (np.diff(values['Z'], axis=2) >= 0).all()
    np.testing.assert_allclose(values['A'], expected_alpha, rtol=tolerance, atol=tolerance)
    assert np.isfinite(values['A']).all() and (values['A'] >= 0).all() and (values['A'] <= 1).all()
    remaining = np.ones((height, width), np.float64)
    flatten = np.zeros((height, width, 3), np.float64)
    coverage = np.zeros((height, width), np.float64)
    for layer in range(4):
        a = values['A'][:, :, layer]
        for c, name in enumerate('RGB'):
            np.testing.assert_array_equal(values[name][:, :, layer], plate[c] * a)
            flatten[:, :, c] += remaining * values[name][:, :, layer]
        coverage += remaining * a
        remaining *= 1 - a.astype(np.float64)
    np.testing.assert_allclose(coverage, 1, rtol=0, atol=2e-15)
    np.testing.assert_allclose(flatten, plate.transpose(1, 2, 0), rtol=2e-7, atol=2e-7)
    return {'depth_max_relative': float(np.max(np.abs(values['Z'] - expected_z) / expected_z)),
            'alpha_max_absolute': float(np.max(np.abs(values['A'] - expected_alpha))),
            'flatten_max_absolute': float(np.max(np.abs(flatten - plate.transpose(1, 2, 0)))),
            'coverage_max_absolute': float(np.max(np.abs(coverage - 1)))}


def admission(command, output):
    source = output / 'models/plumbing-only.onnx'
    rejected = output / 'models/rejected.onnx'
    companion = source.with_suffix('.backbone.onnx')
    shutil.copyfile(companion, rejected.with_suffix('.backbone.onnx'))
    for key, value in [('strata.objective', None),
                       ('strata.objective', 'coverage-bce-logits-v1'),
                       ('strata.layers', '3'), ('strata.backbone_sha256', '0' * 64),
                       ('strata.feature_backbone_sha256', '0' * 64)]:
        graph = onnx.load(source)
        props = {item.key: item.value for item in graph.metadata_props}
        props.pop(key)
        if value is not None:
            props[key] = value
        onnx.helper.set_model_props(graph, props)
        onnx.save(graph, rejected)
        bad = command.copy()
        bad[bad.index('--head') + 1] = str(rejected)
        result = subprocess.run(bad, capture_output=True, text=True)
        assert result.returncode == 1 and 'head metadata mismatch: ' + key in result.stderr, result
    graph = onnx.load(source)
    for node in graph.graph.node:
        for i, name in enumerate(node.output):
            if name == 'inverse_depth':
                node.output[i] = 'inverse_untransposed'
    graph.graph.node.append(onnx.helper.make_node('Transpose', ['inverse_untransposed'],
        ['inverse_depth'], perm=[0, 1, 3, 2]))
    onnx.save(graph, rejected)
    bad = command.copy()
    bad[bad.index('--head') + 1] = str(rejected)
    result = subprocess.run(bad, capture_output=True, text=True)
    assert result.returncode == 1 and 'head output shape mismatch' in result.stderr, result
    for name in (source, companion):
        result = subprocess.run(command + ['--out-depth', str(name)], capture_output=True, text=True)
        assert result.returncode == 2 and result.stderr.startswith('strata depth'), result
    for flag in ('--out-depth', '--out-deep'):
        assert not Path(command[command.index(flag) + 1]).exists()


def runtime_command(cli, output, case, provider):
    return [str(cli.resolve()), 'depth', '--model', str(output / 'models/backbone.onnx'),
        '--head', str(output / 'models/plumbing-only.onnx'), '--in', str(output / (case['id'] + '.exr')),
        '--out-depth', str(output / (case['id'] + '.depth.exr')),
        '--out-deep', str(output / (case['id'] + '.deep.exr')),
        '--provider', provider, '--f-px', str(case['focal_px']),
        '--exposure', str(case['exposure']), '--in-colorspace', case['colorspace']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--provider', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    cases_path = ROOT / 'runtime/tests/parity_cases.json'
    with tempfile.TemporaryDirectory() as temporary:
        output = args.out or Path(temporary)
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(ROOT / 'training/posttrain/.venv/bin/python'),
            str(ROOT / 'tools/layered_oracle.py'), str(output), str(cases_path)], check=True)
        backbone = output / 'models/backbone.onnx'
        write_manifest(backbone)
        rows = []
        for case in json.loads(cases_path.read_text())['layered']:
            with np.load(output / (case['id'] + '.npz')) as archive:
                reference = dict(archive)
            plate_path = output / (case['id'] + '.exr')
            OpenEXR.File({}, {c: reference['plate'][i] for i, c in enumerate('RGB')}).write(str(plate_path))
            deep = output / (case['id'] + '.deep.exr')
            command = runtime_command(args.cli, output, case, args.provider)
            if not rows:
                admission(command, output)
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
            report = json.loads(completed.stdout)
            assert report['provider'] == args.provider and not report['fallback_reason']
            measured = compare(deep, reference, 2e-5 if args.provider == 'cpu' else 2e-4, case['colorspace'])
            rows.append({'id': case['id'], **measured, 'timing': report})
            print(json.dumps(rows[-1]), flush=True)
        (output / 'layered-results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()

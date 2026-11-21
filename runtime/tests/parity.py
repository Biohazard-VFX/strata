#!/usr/bin/env python3
"""Fail on native CUDA/PyTorch parity regressions; never rescale predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import OpenEXR
from PIL import Image

from deep_readback import verify_deep


def relative(native, reference):
    assert native.shape == reference.shape
    assert not np.isnan(native).any()
    np.testing.assert_array_equal(np.isposinf(native), np.isposinf(reference))
    np.testing.assert_array_equal(np.isneginf(native), np.isneginf(reference))
    finite = np.isfinite(reference)
    assert finite.any()
    return np.abs(native[finite]-reference[finite]) / np.maximum(np.abs(reference[finite]),1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--reference-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--max-relative",type=float,default=.05)
    parser.add_argument("--median-relative",type=float,default=.005)
    parser.add_argument("--aggregate-median-relative",type=float,default=.001)
    parser.add_argument("--focal-relative",type=float,default=.001)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())
    assert cases and len({case['id'] for case in cases}) == len(cases)
    args.out.mkdir(parents=True,exist_ok=True)
    report = []
    all_relative = {"camera_z_metres":[],"network_inverse_depth":[],"inverse_metres":[]}
    subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.free","--format=csv"],check=True)
    for index,case in enumerate(cases):
        prefix = args.out / case['id']
        plate = Path(case['input'])
        with plate.open('rb') as source:
            digest = hashlib.file_digest(source,'sha256').hexdigest()
        with np.load(args.reference_dir/(case['id']+'.npz')) as reference:
            assert str(reference['input_sha256']) == digest, 'stale PyTorch oracle'
            assert float(reference['exposure']) == case.get('exposure',0)
            assert bool(reference['auto_focal']) == case.get('auto_focal',False)
            assert float(reference['focal_input']) == (0 if case.get('auto_focal',False) else case['focal_px'])
            command = [str(args.cli),'depth','--model',str(args.model),'--in',str(plate),
                       '--out-depth',str(prefix.with_suffix('.depth.exr')),
                       '--out-deep',str(prefix.with_suffix('.deep.exr')),
                       '--provider','cuda']
            if not case.get('auto_focal',False):
                command += ['--f-px',str(case['focal_px'])]
            command += ['--exposure',str(case.get('exposure',0))]
            completed = subprocess.run(command,text=True,capture_output=True,check=False)
            (prefix.with_suffix('.stderr.log')).write_text(completed.stderr)
            assert completed.returncode == 0, completed.stderr
            timing = json.loads(completed.stdout)
            assert timing['provider'] == 'cuda', 'CPU fallback is not GPU parity'
            assert abs(timing['focal_px']-float(reference['focal_px']))/float(reference['focal_px']) <= args.focal_relative
            row = {'id':case['id'],'timing':timing}
            with OpenEXR.File(str(prefix.with_suffix('.depth.exr')),separate_channels=True) as file:
                assert set(file.channels()) == {'Z','network_inverse_depth','inverse_metres'}
                for field,channel in (('camera_z_metres','Z'),('network_inverse_depth','network_inverse_depth'),('inverse_metres','inverse_metres')):
                    values = file.channels()[channel].pixels.copy()
                    error = relative(values,reference[field])
                    stats = {'max_relative':float(error.max()),'median_relative':float(np.median(error)),
                             'p99_relative':float(np.quantile(error,.99))}
                    row[field] = stats
                    all_relative[field].append(error)
                    assert stats['max_relative'] <= args.max_relative, (case['id'],field,stats)
                    assert stats['median_relative'] <= args.median_relative, (case['id'],field,stats)
                z = file.channels()['Z'].pixels.copy()
            if index == 0:
                rgb = np.asarray(Image.open(plate).convert('RGB'),dtype=np.float32)/255
                verify_deep(prefix.with_suffix('.deep.exr'),rgb,z)
            report.append(row)
            (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    aggregate = {}
    for field,arrays in all_relative.items():
        errors = np.concatenate(arrays)
        aggregate[field] = {'max_relative':float(errors.max()),'median_relative':float(np.median(errors)),
                            'p99_relative':float(np.quantile(errors,.99)),
                            'finite_pixels':int(errors.size)}
    summary = {'cases':len(report),'aggregate':aggregate}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    for field,stats in aggregate.items():
        assert stats['median_relative'] <= args.aggregate_median_relative, (field,stats)


if __name__ == '__main__':
    main()

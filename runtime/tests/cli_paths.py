#!/usr/bin/env python3
"""The CLI rejects outputs that resolve to an input or to another output."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def run(cli, cwd, **paths):
    defaults = {'model': 'model.onnx', 'in': 'input.png', 'out-depth': 'depth.exr', 'out-deep': 'deep.exr'}
    defaults.update({key.replace('_', '-'): value for key, value in paths.items()})
    command = [str(cli), 'depth', '--provider', 'cpu', '--f-px', '1']
    for key, value in defaults.items():
        command += [f'--{key}', str(value)]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def main():
    cli, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    shutil.rmtree(root, ignore_errors=True)
    (root/'sub').mkdir(parents=True)
    os.symlink(root, root/'link')
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(root/'input.png')
    (root/'model.onnx').write_bytes(b'model')
    (root/'camera.json').write_text('{}')
    inputs = {name: (root/name).read_bytes() for name in ('input.png', 'model.onnx', 'camera.json')}
    os.link(root/'input.png', root/'hardlink.png')
    aliases = [
        {'out_depth': './input.png'},
        {'out_depth': f'{root}/./input.png'},
        {'out_deep': 'sub/../input.png'},
        {'out_depth': f'{root}/sub/../model.onnx'},
        {'out_deep': root/'input.png'},
        {'in': root/'input.png', 'out_depth': 'input.png'},
        {'out_depth': 'link/input.png'},
        {'out_deep': 'hardlink.png'},
        {'camera': 'camera.json', 'out_position': 'link/sub/../camera.json'},
        {'out_depth': 'out.exr', 'out_deep': './out.exr'},
        {'out_depth': 'out.exr', 'out_deep': root/'out.exr'},
        {'out_deep': 'deep.exr', 'out_position': 'link/deep.exr'},
        {'out_depth': 'sub/../depth.exr', 'out_position': 'link/sub/../depth.exr'},
        {'out_depth': 'missing/depth.exr'},
        {'out_depth': 'sub/'},
    ]
    for paths in aliases:
        completed = run(cli, root, **paths)
        assert completed.returncode == 2 and completed.stderr.startswith('strata depth'), (paths, completed)
        for name, content in inputs.items():
            assert (root/name).read_bytes() == content, (paths, name)
    for paths in ({}, {'out_depth': 'link/sub/depth.exr', 'out_position': 'P.exr'}, {'out_deep': root/'sub/deep.exr'}):
        completed = run(cli, root, **paths)
        assert completed.returncode == 1 and 'model contract missing' in completed.stderr, (paths, completed)
    print(f'{len(aliases)} aliased output paths rejected without touching inputs')


if __name__ == '__main__':
    main()

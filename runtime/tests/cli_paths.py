#!/usr/bin/env python3
"""The CLI rejects outputs inside the model directory or resolving to an input, a model file or another output."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from strata_contract import write_manifest


def run(cli, cwd, **paths):
    defaults = {'model': 'models/model.onnx', 'in': 'input.png', 'out-depth': 'depth.exr', 'out-deep': 'deep.exr'}
    defaults.update({key.replace('_', '-'): value for key, value in paths.items()})
    command = [str(cli), 'depth', '--provider', 'cpu', '--f-px', '1']
    for key, value in defaults.items():
        command += [f'--{key}', str(value)]
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False, timeout=120)


def main():
    cli, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    shutil.rmtree(root, ignore_errors=True)
    (root/'sub').mkdir(parents=True)
    (root/'models').mkdir()
    os.symlink(root, root/'link')
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(root/'input.png')
    (root/'models/model.onnx').write_bytes(b'model')
    (root/'camera.json').write_text('{}')
    inputs = {name: (root/name).read_bytes() for name in ('input.png', 'models/model.onnx', 'camera.json')}
    os.link(root/'input.png', root/'hardlink.png')
    aliases = [
        {'out_depth': './input.png'},
        {'out_depth': f'{root}/./input.png'},
        {'out_deep': 'sub/../input.png'},
        {'out_depth': f'{root}/sub/../models/model.onnx'},
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
    model_directory(cli, root/'directory')


def depth_model(path, location='depth.weights'):
    image = helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 'h', 'w'])
    output = helper.make_tensor_value_info('network_inverse_depth', TensorProto.FLOAT, [1, 1, 'h', 'w'])
    scale = numpy_helper.from_array(np.array(1, np.float32), 'scale')
    graph = helper.make_graph([
        helper.make_node('Constant', [], ['axes'], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node('ReduceMean', ['image', 'axes'], ['mean'], keepdims=1),
        helper.make_node('Abs', ['mean'], ['positive']),
        helper.make_node('Mul', ['positive', 'scale'], ['network_inverse_depth']),
    ], 'depth', [image], [output], [scale])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 23)], ir_version=10)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location='depth.weights', size_threshold=0)
    if location != 'depth.weights':
        model = onnx.load_model(str(path), load_external_data=False)
        entry = next(item for item in model.graph.initializer[0].external_data if item.key == 'location')
        entry.value = location
        path.write_bytes(model.SerializeToString())
    write_manifest(path)


def tree(directory):
    return {str(path.relative_to(directory)): path.read_bytes() if path.is_file() and not path.is_symlink() else None
            for path in sorted(directory.rglob('*'))}


def model_directory(cli, root):
    """No output may land in the model directory; ORT keeps external data there; an unused FOV is never opened."""
    model, work, outside = root/'model', root/'work', root/'outside'
    (model/'sub').mkdir(parents=True)
    work.mkdir()
    outside.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(work/'input.png')
    depth_model(model/'model.onnx')
    (model/'model.fov.onnx').write_bytes(b'\x0a\x05ab')
    os.symlink(model, work/'link')
    os.symlink(model/'model.onnx', outside/'linked.onnx')
    os.link(model/'model.json', work/'manifest.json')
    os.link(model/'model.fov.onnx', work/'fov.onnx')
    before = tree(model)
    flags = ('out_depth', 'out_deep', 'out_position')
    refused = [
        (work, '../model/model.onnx', '../model/new.exr'),
        (work, '../model/model.onnx', '../model/./depth.weights'),
        (work, '../model/model.onnx', str(model/'model.json')),
        (work, '../model/model.onnx', '../model/sub/../model.fov.onnx'),
        (work, '../model/model.onnx', '../model/sub/new.exr'),
        (work, '../model/model.onnx', 'link/new.exr'),
        (work, '../model/model.onnx', 'link/sub/depth.weights'),
        (work, str(outside/'linked.onnx'), '../model/new.exr'),
        (outside, 'linked.onnx', 'new.exr'),
        (work, '../model/model.onnx', 'manifest.json'),
        (work, '../model/model.onnx', 'fov.onnx'),
        (model, 'model.onnx', 'new.exr'),
        (model, 'model.onnx', './new.exr'),
        (model, './model.onnx', '../model/new.exr'),
        (model/'sub', '../model.onnx', '.'),
        (model/'sub', '../model.onnx', 'new.exr'),
    ]
    for index, (cwd, path, output) in enumerate(refused):
        flag = flags[index % 3]
        completed = run(cli, cwd, model=path, **{'in': work/'input.png', 'out_depth': work/'depth.exr', 'out_deep': work/'deep.exr', flag: output})
        assert completed.returncode == 2 and completed.stderr.startswith('strata depth'), (cwd, path, output, completed)
        assert tree(model) == before, (cwd, path, output)
    outputs = {'out_depth': 'depth.weights', 'out_deep': 'model.json', 'out_position': 'model.fov.onnx'}
    for fov in ('corrupt', 'fifo', 'missing'):
        if fov == 'fifo':
            (model/'model.fov.onnx').unlink()
            os.mkfifo(model/'model.fov.onnx')
        elif fov == 'missing':
            (model/'model.fov.onnx').unlink()
        completed = run(cli, work, model='../model/model.onnx', **outputs)
        assert completed.returncode == 0, (fov, completed)
        assert all((work/name).stat().st_size > 0 for name in outputs.values()), fov
        for name in outputs.values():
            (work/name).unlink()
        assert {name: content for name, content in tree(model).items() if name != 'model.fov.onnx'} == \
            {name: content for name, content in before.items() if name != 'model.fov.onnx'}, fov
    print(f'{len(refused)} outputs in or aliasing the model directory refused; unused corrupt, FIFO and missing FOV ignored')
    escapes(cli, root/'escape')


def escapes(cli, root):
    """The pinned ONNX Runtime refuses external data outside the model directory, which the directory rule relies on."""
    (root/'model').mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(root/'input.png')
    (root/'outside.bin').write_bytes(np.array(1, np.float32).tobytes())
    os.symlink(root, root/'model/up')
    locations = ['../outside.bin', str(root/'outside.bin'), 'up/outside.bin']
    for index, location in enumerate(locations):
        path = root/'model'/f'model{index}.onnx'
        depth_model(path, location)
        completed = run(cli, root, model=path, out_depth=f'depth{index}.exr', out_deep=f'deep{index}.exr')
        reason = 'Absolute path not allowed' if location.startswith('/') else 'External data path escapes model directory'
        assert completed.returncode == 1 and reason in completed.stderr, (location, completed)
    print(f'{len(locations)} external data locations outside the model directory refused by ONNX Runtime')


if __name__ == '__main__':
    main()

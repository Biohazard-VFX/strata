#!/usr/bin/env python3
"""The CLI rejects outputs that resolve to an input, a model file or another output."""
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
    model_dependencies(cli, root/'model')


def external(location):
    tensor = TensorProto(name=location, data_type=TensorProto.FLOAT, dims=[1], data_location=TensorProto.EXTERNAL)
    tensor.external_data.add(key='location', value=location)
    return tensor


def sparse(prefix):
    tensor = onnx.SparseTensorProto(dims=[1])
    tensor.values.CopyFrom(external(f'{prefix}-values.bin'))
    tensor.indices.CopyFrom(external(f'{prefix}-indices.bin'))
    return tensor


def subgraph(location):
    return helper.make_graph([], location, [], [], [external(location)])


def attribute(name, kind, **value):
    proto = onnx.AttributeProto(name=name, type=kind)
    for key, item in value.items():
        if isinstance(item, list):
            getattr(proto, key).extend(item)
        else:
            getattr(proto, key).CopyFrom(item)
    return proto


def fov_model(absolute):
    """Place an external tensor in every message that can hold one; the FOV is never loaded here."""
    node = helper.make_node('Custom', [], ['out'], alpha=.5)
    node.attribute.extend([
        attribute('t', onnx.AttributeProto.TENSOR, t=external('attribute.bin')),
        attribute('tensors', onnx.AttributeProto.TENSORS, tensors=[external('attribute-list.bin')]),
        attribute('g', onnx.AttributeProto.GRAPH, g=subgraph('subgraph.bin')),
        attribute('graphs', onnx.AttributeProto.GRAPHS, graphs=[subgraph('subgraph-list.bin')]),
        attribute('sparse', onnx.AttributeProto.SPARSE_TENSOR, sparse_tensor=sparse('attribute-sparse')),
        attribute('sparses', onnx.AttributeProto.SPARSE_TENSORS, sparse_tensors=[sparse('attribute-sparse-list')]),
    ])
    graph = helper.make_graph([node], 'fov', [], [], [external('initializer.bin'), external(absolute)])
    graph.sparse_initializer.append(sparse('sparse-initializer'))
    function = onnx.FunctionProto(name='f', domain='custom')
    function.node.append(helper.make_node('Constant', [], ['c'], value=external('function.bin')))
    function.attribute_proto.append(attribute('default', onnx.AttributeProto.TENSOR, t=external('function-default.bin')))
    model = helper.make_model(graph)
    model.functions.append(function)
    return model


def depth_model(path):
    image = helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 'h', 'w'])
    output = helper.make_tensor_value_info('network_inverse_depth', TensorProto.FLOAT, [1, 1, 'h', 'w'])
    graph = helper.make_graph([
        helper.make_node('Constant', [], ['axes'], value=numpy_helper.from_array(np.array([1], np.int64))),
        helper.make_node('ReduceMean', ['image', 'axes'], ['mean'], keepdims=1),
        helper.make_node('Abs', ['mean'], ['positive']),
        helper.make_node('Mul', ['positive', 'scale'], ['network_inverse_depth']),
    ], 'depth', [image], [output], [numpy_helper.from_array(np.array(1, np.float32), 'scale')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 23)], ir_version=10)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location='depth.weights', size_threshold=0)
    write_manifest(path)


def model_dependencies(cli, root):
    root.mkdir()
    (root/'sub').mkdir()
    os.symlink(root, root/'link')
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(root/'input.png')
    depth_model(root/'model.onnx')
    absolute = root/'absolute.bin'
    (root/'model.fov.onnx').write_bytes(fov_model(str(absolute)).SerializeToString())
    names = ['model.onnx', 'model.json', 'model.fov.onnx', 'depth.weights', 'initializer.bin', 'absolute.bin',
             'sparse-initializer-values.bin', 'sparse-initializer-indices.bin', 'attribute.bin', 'attribute-list.bin',
             'subgraph.bin', 'subgraph-list.bin', 'attribute-sparse-values.bin', 'attribute-sparse-indices.bin',
             'attribute-sparse-list-values.bin', 'attribute-sparse-list-indices.bin', 'function.bin', 'function-default.bin']
    for name in names:
        if not (root/name).exists():
            (root/name).write_bytes(name.encode())
    files = {name: (root/name).read_bytes() for name in names}
    flags = ('out_depth', 'out_deep', 'out_position')
    count = 0
    for name in names:
        for alias in (name, f'link/{name}', f'sub/../{name}', root/name):
            completed = run(cli, root, **{flags[count % 3]: alias})
            count += 1
            assert completed.returncode == 2 and completed.stderr.startswith('strata depth'), (alias, completed)
            for other, content in files.items():
                assert (root/other).read_bytes() == content, (alias, other)
    completed = run(cli, root, out_position='P.exr')
    assert completed.returncode == 0, completed
    assert all((root/name).read_bytes() == content for name, content in files.items())
    (root/'model.fov.onnx').write_bytes(b'\x0a\x05ab')
    completed = run(cli, root, out_depth='other.exr')
    assert completed.returncode == 1 and 'malformed ONNX model' in completed.stderr, completed
    print(f'{count} model dependency outputs rejected across {len(names)} files without touching them')


if __name__ == '__main__':
    main()

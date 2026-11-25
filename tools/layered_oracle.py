import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
from torch.nn import functional as functional

from strata_training.data.encoder_input import display, normalize
from strata_training.eval.predict import load_model
from strata_training.model.export import export_head, smoke_checkpoint


def fixture(path):
    weights = {
        'features_weight': np.full((128, 3, 1, 1), 1 / 384, np.float32),
        'features_bias': np.arange(128, dtype=np.float32) / 1280,
        'inverse_weight': np.full((32, 128, 1, 1), 1 / 128, np.float32),
        'inverse_bias': np.full(32, .2, np.float32),
        'final_weight': np.full((1, 32, 1, 1), 1 / 32, np.float32),
        'final_bias': np.array([.3], np.float32),
    }
    shape = [1, 128, 'h', 'w']
    nodes = [
        helper.make_node('Conv', ['image', 'features_weight', 'features_bias'], ['features']),
        helper.make_node('Conv', ['features', 'inverse_weight', 'inverse_bias'], ['low']),
        helper.make_node('Relu', ['low'], ['positive']),
        helper.make_node('Cast', ['positive'], ['full'], to=TensorProto.FLOAT),
        helper.make_node('Conv', ['full', 'final_weight', 'final_bias'], ['inverse']),
        helper.make_node('Relu', ['inverse'], ['network_inverse_depth']),
    ]
    graph = helper.make_graph(nodes, 'fixed-feature-glue',
        [helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, 'h', 'w'])],
        [helper.make_tensor_value_info('network_inverse_depth', TensorProto.FLOAT, [1, 1, 'h', 'w'])],
        [numpy_helper.from_array(value, name) for name, value in weights.items()],
        value_info=[helper.make_tensor_value_info('features', TensorProto.FLOAT, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 23)], ir_version=10)
    onnx.save(model, path)
    return {name: torch.from_numpy(value) for name, value in weights.items()}


def main():
    output, cases_path = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'models').mkdir(exist_ok=True)
    backbone = output / 'models/backbone.onnx'
    weights = fixture(backbone)
    checkpoint = output / 'models/plumbing-only.pt'
    smoke_checkpoint(checkpoint)
    original = backbone.read_bytes()
    exported = output / 'models/plumbing-only.onnx'
    export_head(checkpoint, exported, backbone)
    export_head(checkpoint, output / 'models/plumbing-only-repeat.onnx', backbone)
    assert exported.read_bytes() == (output / 'models/plumbing-only-repeat.onnx').read_bytes()
    assert backbone.read_bytes() == original
    source = onnx.load(backbone, load_external_data=False)
    companion = onnx.load(exported.with_suffix('.backbone.onnx'), load_external_data=False)
    assert list(source.graph.node) == list(companion.graph.node[:-1])
    assert list(source.graph.initializer) == list(companion.graph.initializer)
    metadata = {item.key: item.value for item in onnx.load(exported).metadata_props}
    assert metadata['strata.purpose'] == 'plumbing-only'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['config']['purpose'] == 'plumbing-only' and state['config']['trained'] is False
    state['config']['objective'] = 'coverage-bce-logits-v1'
    incompatible = output / 'models/incompatible.pt'
    torch.save(state, incompatible)
    try:
        export_head(incompatible, output / 'models/rejected.onnx', backbone)
    except ValueError as error:
        assert 'coverage-by-construction' in str(error), error
    else:
        raise AssertionError('Incompatible checkpoint objective was accepted')
    model, _ = load_model(checkpoint, 'cpu')
    cases = json.loads(cases_path.read_text())['layered']
    for case in cases:
        height, width = case['height'], case['width']
        rng = np.random.default_rng(case['seed'])
        plate = rng.uniform(-.1, 3, (3, height, width)).astype(np.float32)
        image = torch.from_numpy(plate)[None]
        encoded = normalize(display(image, 'linear-sRGB', case['exposure']))
        features = functional.conv2d(encoded, weights['features_weight'], weights['features_bias'])
        base = functional.conv2d(features, weights['inverse_weight'], weights['inverse_bias']).relu()
        base = functional.conv2d(base, weights['final_weight'], weights['final_bias']).relu()
        base = base * (width / case['focal_px'])
        with torch.no_grad():
            prediction = model(features, image, base)
        np.savez(output / (case['id'] + '.npz'), plate=plate,
            inverse=prediction['inverse_depth'][0].numpy(), alpha=prediction['alpha'][0].numpy())


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()

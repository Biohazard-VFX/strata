#!/usr/bin/env python3
"""Cross-language contract gate: generated constants plus real C load rejection."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import struct

from strata_contract import c_header, spec, srgb_encode, srgb_samples, validate_manifest, write_manifest

ROOT = Path(__file__).resolve().parent.parent


def run(cli, directory, expected):
    result = subprocess.run([str(cli),'depth','--model',str(directory/'model.onnx'),
                             '--in',str(directory/'input.png'),
                             '--out-depth',str(directory/'depth.exr'),
                             '--out-deep',str(directory/'deep.exr'),
                             '--provider','cpu','--f-px','1'],capture_output=True,text=True,check=False)
    assert result.returncode == 1, result.stdout+result.stderr
    assert expected in result.stderr, result.stderr


def main():
    assert (ROOT/'contract/strata_contract.h').read_text() == c_header(), 'regenerate contract C header'
    if len(sys.argv) == 1:
        print('generated C contract matches canonical version-one JSON')
        return
    import numpy as np
    import onnx
    from PIL import Image

    cli,directory = Path(sys.argv[1]),Path(sys.argv[2])
    directory.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(directory/'input.png')
    manifest = directory/'model.json'
    manifest.unlink(missing_ok=True)
    run(cli,directory,'model contract missing')
    model = directory/'model.onnx'
    model.unlink(missing_ok=True)
    write_manifest(model)
    validate_manifest(manifest)
    run(cli,directory,'ONNX Runtime:')
    changes = [
        (('contract_version',),2),
        (('input','name'),'wrong'),
        (('input','dtype'),'float16'),
        (('input','shape'),[1,4,'height','width']),
        (('preprocessing','mean'),[0,0,0]),
        (('preprocessing','multiple'),16),
        (('postprocessing','camera_z_metres'),'incorrect'),
        (('fov','output_name'),'wrong'),
        (('deep','channels'),['Z']),
        (('unknown',),True),
    ]
    for path,value in changes:
        data = copy.deepcopy(spec())
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        manifest.write_text(json.dumps(data))
        try:
            validate_manifest(manifest)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Python accepted incompatible manifest: {path}')
        run(cli,directory,'incompatible model contract')
    for malformed in ('{',json.dumps(spec())+' junk'):
        manifest.write_text(malformed)
        run(cli,directory,'incompatible model contract')
    write_manifest(model)
    for name,channels in (('wrong',1),(spec()['output']['name'],3),(spec()['output']['name'],1)):
        image = onnx.helper.make_tensor_value_info('image',onnx.TensorProto.FLOAT,[1,3,'h','w'])
        output = onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,channels,'h','w'])
        axes = onnx.helper.make_tensor('axes',onnx.TensorProto.INT64,[1],[1])
        nodes = [onnx.helper.make_node('ReduceMean',['image','axes'],['mean'],keepdims=1),
                 onnx.helper.make_node('Abs',['mean'],[name])] if channels == 1 else [onnx.helper.make_node('Identity',['image'],[name])]
        graph = onnx.helper.make_graph(nodes,'contract_fixture',[image],[output],initializer=[axes] if channels == 1 else [])
        fixture = onnx.helper.make_model(graph,opset_imports=[onnx.helper.make_opsetid('',23)])
        fixture.ir_version = 10
        onnx.save(fixture,str(model))
        if name != spec()['output']['name'] or channels != 1:
            run(cli,directory,'ONNX tensor contract mismatch')
    command = [str(cli),'depth','--model',str(model),'--in',str(directory/'input.png'),
               '--out-depth',str(directory/'depth.exr'),'--out-deep',str(directory/'deep.exr'),
               '--out-position',str(directory/'P.exr'),'--provider','cpu']
    import OpenEXR
    for option,value in (('--f-px','nan'),('--f-px','-1'),('--f-px','inf'),('--repeat','2'),('--device','1.5'),('--device','-1'),('--exposure','nan'),('--out-position',str(directory/'input.png'))):
        invalid = subprocess.run(command+[option,value],capture_output=True,text=True,check=False)
        assert invalid.returncode == 2 and invalid.stderr.startswith('strata depth'), (option,value,invalid.returncode,invalid.stderr)
    for world in (False,True):
        camera = {'version':1,'fx_px':1,'fy_px':2,'cx_px':1,'cy_px':1,
                  'sensor_width_mm':36,'sensor_height_mm':24,
                  'camera_to_world':[0,-1,0,10,1,0,0,20,0,0,1,30,0,0,0,1]}
        camera_path = directory/'camera.json'
        camera_path.write_text(json.dumps(camera))
        completed = subprocess.run(command+(['--camera',str(camera_path)] if world else ['--f-px','1']),capture_output=True,text=True,check=False)
        assert completed.returncode == 0, completed.stdout+completed.stderr
        report = json.loads(completed.stdout)
        assert report['provider'] == 'cpu' and report['fallback_reason'] == '' and report['focal_source'] == 'provided'
        assert set(report['stages']) == {'prepare_ms','focal_ms','upload_ms','run_ms','download_ms','postprocess_ms'}
        with OpenEXR.File(str(directory/'depth.exr'),separate_channels=True) as file:
            assert set(file.channels()) == {'Z','network_inverse_depth'}
            z = file.channels()['Z'].pixels
            q = abs(np.mean(-np.array(spec()['preprocessing']['mean'],dtype=np.float32)/np.array(spec()['preprocessing']['std'],dtype=np.float32)))
            np.testing.assert_allclose(file.channels()['network_inverse_depth'].pixels,q,rtol=1e-6)
            np.testing.assert_allclose(z,1/(2*q),rtol=1e-6)
            if world:
                assert file.header()['strata:fy_px'] == 2
                assert file.header()['strata:sensor_width_mm'] == 36
        y,x = np.mgrid[:2,:2].astype(np.float32)
        px = (x+.5-1)*z
        py = (y+.5-1)*z/(2 if world else 1)
        expected_p = np.stack((10-py,20+px,30+z) if world else (px,py,z),-1)
        with OpenEXR.File(str(directory/'P.exr'),separate_channels=True) as file:
            assert set(file.channels()) == {'P.x','P.y','P.z'}
            values = np.stack([file.channels()['P.'+c].pixels for c in 'xyz'],-1)
            np.testing.assert_allclose(values,expected_p,rtol=1e-6)
            assert file.header()['strata:P_space'] == ('world' if world else 'camera')

    def save_fov(values):
        angles = onnx.helper.make_tensor('angles',onnx.TensorProto.FLOAT,[1,2],values)
        zero = onnx.helper.make_tensor('zero',onnx.TensorProto.FLOAT,[],[0])
        one = onnx.helper.make_tensor('one',onnx.TensorProto.INT64,[2],[0,1])
        fov_graph = onnx.helper.make_graph(
            [onnx.helper.make_node('ReduceMean',['image'],['mean'],keepdims=0),
             onnx.helper.make_node('Mul',['mean','zero'],['zeroed']),
             onnx.helper.make_node('Unsqueeze',['zeroed','one'],['column']),
             onnx.helper.make_node('Add',['column','angles'],[spec()['fov']['output_name']])],
            'fov_fixture',[image],
            [onnx.helper.make_tensor_value_info(spec()['fov']['output_name'],onnx.TensorProto.FLOAT,[1,2])],
            initializer=[angles,zero,one])
        fov = onnx.helper.make_model(fov_graph,opset_imports=[onnx.helper.make_opsetid('',23)])
        fov.ir_version = 10
        onnx.save(fov,str(directory/'model.fov.onnx'))
    for values in ([50,float('nan')],[50,float('inf')],[50,float('-inf')],[float('nan'),30],
                   [50,0],[50,-10],[50,180],[50,200],[0,30],[180,30]):
        save_fov(values)
        for path in ('depth.exr','deep.exr','P.exr'):
            (directory/path).unlink(missing_ok=True)
        rejected = subprocess.run(command,capture_output=True,text=True,check=False)
        assert rejected.returncode == 1 and 'invalid estimated field of view' in rejected.stderr, (values,rejected.returncode,rejected.stdout,rejected.stderr)
        assert not rejected.stdout and not (directory/'depth.exr').exists(), values
    save_fov([50,30])
    estimated = subprocess.run(command,capture_output=True,text=True,check=False)
    assert estimated.returncode == 0, estimated.stderr
    report = json.loads(estimated.stdout)
    assert report['focal_source'] == 'estimated' and abs(report['focal_px']-1/np.tan(np.radians(15))) < 1e-4, report
    bad = directory/'bad.onnx'
    onnx.save(onnx.helper.make_model(onnx.helper.make_graph([onnx.helper.make_node('Identity',['image'],[name])],'bad',[image],
        [onnx.helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,[1,3,'h','w'])]),opset_imports=[onnx.helper.make_opsetid('',23)],ir_version=10),str(bad))
    write_manifest(bad)
    if len(sys.argv) > 3:
        subprocess.run([sys.argv[3],str(model),str(bad),str(directory/'missing.onnx')],check=True)
    if len(sys.argv) > 4:
        bits = lambda value: struct.unpack('<I',struct.pack('<f',value))[0]
        vectors = ''.join(f'{bits(value):08x} {bits(srgb_encode(value)):08x}\n' for value in srgb_samples())
        subprocess.run([sys.argv[4]],input=vectors,text=True,check=True)
    fixture.graph.node[-1].output[0] = 'untransposed_q'
    fixture.graph.node.append(onnx.helper.make_node('Transpose',['untransposed_q'],[name],perm=[0,1,3,2]))
    onnx.save(fixture,str(model))
    Image.fromarray(np.zeros((3,65,3),dtype=np.uint8)).save(directory/'input.png')
    completed = subprocess.run(command+['--f-px','1'],capture_output=True,text=True,check=False)
    assert completed.returncode == 1 and "output 'network_inverse_depth'" in completed.stderr, completed.stdout+completed.stderr
    print('Python/C contract rejection, real CPU inference, depth units, lifecycle and camera/world P passed')


if __name__ == '__main__':
    main()

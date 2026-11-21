#!/usr/bin/env python3
"""Exercise public C IO in a native process, then independently read every sample."""
import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import OpenEXR
from PIL import Image


def verify_deep(path, rgb, z):
    expected = {"R":rgb[...,0], "G":rgb[...,1], "B":rgb[...,2],
                "A":np.ones_like(z), "Z":z, "ZBack":z}
    with OpenEXR.File(str(path), separate_channels=True) as file:
        assert file.header()["type"] == OpenEXR.deepscanline
        assert file.header()["strata:rgb_association"] == "premultiplied"
        assert file.header()["strata:alpha_semantics"] == "coverage"
        channels = file.channels()
        assert set(channels) == set(expected)
        for name, target in expected.items():
            array = channels[name].pixels
            assert array.shape == z.shape
            assert all(sample is not None and len(sample) == 1 for sample in array.flat)
            values = np.fromiter((sample[0] for sample in array.flat),np.float32,count=z.size).reshape(z.shape)
            np.testing.assert_array_equal(values, target)
    np.savez(path.with_suffix(".expected.npz"), counts=np.ones(z.shape,dtype=np.uint32),
             **{name:values.reshape(-1) for name,values in expected.items()})
    subprocess.run(["/usr/bin/python3",str(Path(__file__).with_name("oiio_check.py")),str(path)],
                   check=True,env={**os.environ,"PYTHONPATH":""})


def verify_variable(path, empty=False):
    counts = np.array([[0,3,1],[2,1,0]],dtype=np.uint32)
    rgba = np.arange(28,dtype=np.float32).reshape(7,4)-10
    rgba[:,3] = 1
    rgba[1,:3] *= .25
    rgba[1,3] = .25
    z = np.array([3,1,2,5,6,4,np.inf],dtype=np.float32)
    back = np.array([3.5,1.5,2.5,5.5,6.5,4.5,np.inf],dtype=np.float32)
    expected = {name:rgba[:,c] for c,name in enumerate('RGBA')}
    expected.update(Z=z,ZBack=back)
    if empty:
        counts.fill(0)
        expected = {name:values[:0] for name,values in expected.items()}
    with OpenEXR.File(str(path),separate_channels=True) as file:
        assert file.header()['type'] == OpenEXR.deepscanline
        assert set(file.channels()) == set(expected)
        for name,values in expected.items():
            offset = 0
            for samples,count in zip(file.channels()[name].pixels.flat,counts.flat,strict=True):
                if count == 0:
                    assert samples is None or len(samples) == 0
                else:
                    assert len(samples) == count
                    np.testing.assert_array_equal(samples,values[offset:offset+count])
                offset += int(count)
            assert offset == len(values)
    np.savez(path.with_suffix('.expected.npz'),counts=counts,**expected)
    subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('oiio_check.py')),str(path)],
                   check=True,env={**os.environ,'PYTHONPATH':''})


def main():
    out = Path(sys.argv[2])
    out.mkdir(parents=True,exist_ok=True)
    h,w = 19,7
    rgba = np.arange(h*w*4,dtype=np.float32).reshape(h,w,4)/31-1
    z = np.arange(h*w,dtype=np.float32).reshape(h,w)+.25
    z[2,3] = np.inf
    inverse = (1000/(w*z)).astype(np.float32)
    with np.errstate(divide='ignore'):
        derived = np.float32(1000)/(np.float32(w)*inverse)
    header = {"type":OpenEXR.scanlineimage,"compression":OpenEXR.ZIP_COMPRESSION,
              "dataWindow":(np.array([-3,9],dtype=np.int32),np.array([w-4,h+8],dtype=np.int32))}
    OpenEXR.File(header,{c:rgba[...,i].copy() for i,c in enumerate("RGB")}).write(str(out/"input.exr"))
    for extension in ("png","jpg"):
        Image.fromarray(np.full((2,2,3),128,dtype=np.uint8)).save(out/f"input.{extension}")
    (out/'camera.json').write_text(json.dumps({'version':1,'fx_px':2,'fy_px':4,'cx_px':1,'cy_px':1,
        'camera_to_world':[0,-1,0,10,1,0,0,20,0,0,1,30,0,0,0,1]}))
    subprocess.run([sys.argv[1],str(out)],check=True)
    with OpenEXR.File(str(out/"depth.exr"),separate_channels=True) as file:
        assert set(file.channels()) == {"Z","network_inverse_depth"}
        np.testing.assert_array_equal(file.channels()["Z"].pixels,derived)
        np.testing.assert_array_equal(file.channels()["network_inverse_depth"].pixels,inverse)
        assert file.header()["strata:fx_px"] == 1000
        assert file.header()["strata:fy_px"] == 1000
        assert file.header()["strata:frame_index"] == "1001"
        assert file.header()["strata:timestamp_seconds"] == 1
        assert file.header()["strata:sensor_width_mm"] == 36
        assert file.header()["strata:focal_source"] == "provided"
        assert "strata:focal_px" not in file.header() and "strata:sensor_used" not in file.header()
    expected_camera = np.array([[[-.5,-.25,2],[.5,-.25,2]],
                                [[-.5,.25,2],[.5,.25,2]]],dtype=np.float32)
    expected_world = np.array([[[10.25,19.5,32],[10.25,20.5,32]],
                               [[9.75,19.5,32],[9.75,20.5,32]]],dtype=np.float32)
    for space,expected_p in (('camera',expected_camera),('world',expected_world)):
        with OpenEXR.File(str(out/f'P-{space}.exr'),separate_channels=True) as file:
            assert set(file.channels()) == {'P.x','P.y','P.z'}
            points = np.stack([file.channels()[f'P.{c}'].pixels for c in 'xyz'],-1)
            np.testing.assert_array_equal(points,expected_p)
            assert file.header()['strata:P_space'] == space
            assert file.header()['strata:depth_units'] == 'metres'
    verify_deep(out/"deep.exr",rgba[...,:3],derived)
    verify_variable(out/"variable.exr")
    verify_variable(out/"empty.exr",empty=True)
    print("C ABI / PNG / JPEG / offset EXR / flat / deep / OpenEXR / OIIO passed")


if __name__ == "__main__":
    main()

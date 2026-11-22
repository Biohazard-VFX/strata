#!/usr/bin/env python3
"""Independently compare every deep count and channel value with OpenImageIO."""
import sys
from pathlib import Path
import numpy as np
import OpenImageIO as oiio

path = Path(sys.argv[1])
image = oiio.ImageInput.open(str(path))
assert image, oiio.geterror()
spec = image.spec()
assert spec.deep
assert set(spec.channelnames) == {"R","G","B","A","Z","ZBack"}
data = image.read_native_deep_image()
assert data, image.geterror()
assert data.pixels == spec.width * spec.height
with np.load(path.with_suffix('.expected.npz')) as archive:
    counts = archive['counts'].reshape(-1)
    expected = {name:archive[name].reshape(-1) for name in spec.channelnames}
assert len(counts) == data.pixels
offset = 0
for pixel in range(data.pixels):
    count = int(counts[pixel])
    assert data.samples(pixel) == count
    for sample in range(count):
        for channel,name in enumerate(spec.channelnames):
            value = data.deep_value(pixel,channel,sample)
            target = expected[name][offset+sample]
            assert value == target, (pixel,sample,name,value,target)
    offset += count
assert all(offset == len(values) for values in expected.values())
image.close()
print(f"OIIO {oiio.VERSION_STRING}: {data.pixels} pixels, {offset} samples, all counts and values exact")

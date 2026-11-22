import hashlib
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import torch

from strata_training.augment.deep import DeepSamples, save_samples
from strata_training.data.raw_tensor import MappedSamples, mapped_samples


def test_mapped_raw_transport_and_native_column_gather(tmp_path: Path, math_device: str) -> None:
    rng = np.random.default_rng(1901)
    shape = (6, 11)
    counts = rng.integers(0, 9, np.prod(shape), dtype=np.int64)
    offsets = np.concatenate(([0], counts.cumsum()))
    n = int(offsets[-1])
    source = DeepSamples(
        shape,
        offsets,
        np.arange(n, dtype=np.float32) + 1,
        rng.uniform(0, 1, n).astype(np.float32),
        rng.uniform(0, 1, (n, 3)).astype(np.float32),
        rng.integers(0, 4, n, dtype=np.int64),
    )
    root = tmp_path / "deep"
    save_samples(root, source)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    for top, left, height, width in ((0, 0, 6, 11), (1, 3, 4, 5)):
        raw = mapped_samples(root, top, left, height, width)
        encoded = pickle.dumps(raw)
        assert len(encoded) < 1024
        received = cast(MappedSamples, pickle.loads(encoded))
        expected = source.crop(top, left, height, width)
        actual = received.transfer(lambda value: value.to(math_device))
        assert actual.shape == expected.shape
        for key in ("counts", "z", "alpha", "rgb", "ids"):
            np.testing.assert_array_equal(
                getattr(actual, key).cpu().numpy(), getattr(expected, key)
            )
        raw.rgb.zero_()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    assert before == after
    assert torch.isfinite(actual.rgb).all()

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from strata_training.data.cache import display_rgb
from strata_training.data.dataset import Frame, batch
from strata_training.data.metadata import CONTRACT
from strata_training.io import save_arrays, write_json


def test_native_denoised_extraction_persists_matching_inference_image_and_rejects_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features: Any = import_module("strata_training.model.features")
    cache = tmp_path / "target.npz"
    denoised = np.full((3, 32, 32), 0.3, np.float32)
    alpha = np.zeros((4, 32, 32), np.float32)
    alpha[0] = 1
    save_arrays(
        cache,
        alpha=alpha,
        inverse_depth=alpha * 0.5,
        rgb=np.zeros((4, 3, 32, 32)),
        beauty=denoised * 2,
        denoised=denoised,
        srgb=np.full((3, 32, 32), 255, np.uint8),
        mixed=np.zeros((32, 32), bool),
    )
    write_json(
        tmp_path / "camera.json",
        {
            "contract": CONTRACT,
            "frames": [
                {
                    "frame": 1,
                    "width": 32,
                    "height": 32,
                    "fx": 32,
                }
            ],
        },
    )
    write_json(tmp_path / "manifest.json", {"contract": CONTRACT})
    entry = {
        "id": "frame",
        "shot": "shot",
        "frame": 1,
        "split": "train",
        "cache": str(cache),
        "manifest": str(tmp_path / "manifest.json"),
    }
    index = tmp_path / "index.json"
    write_json(index, [entry])
    observed: list[torch.Tensor] = []

    def pyramid(image: torch.Tensor) -> list[torch.Tensor]:
        observed.append(image.clone())
        return [image[:, :1, ::s, ::s] for s in (4, 8, 16, 32)]

    def decode(pyramid: list[torch.Tensor]) -> torch.Tensor:
        return pyramid[0].repeat_interleave(4, -2).repeat_interleave(4, -1).abs() + 1

    model = SimpleNamespace(
        pyramid=pyramid, decode=decode, deep_head=SimpleNamespace(inverse_head=torch.nn.Identity())
    )

    def load(preset: str, device: str) -> SimpleNamespace:
        del preset, device
        return model

    monkeypatch.setattr(features, "_load", load)
    root = tmp_path / "features"
    features.extract(index, root, "depthpro2-s", device="cpu")
    display = display_rgb(denoised, "ACEScg").astype(np.float32) / 255
    expected = (display - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array(
        [0.229, 0.224, 0.225]
    )[:, None, None]
    assert_allclose(observed[0][0].numpy(), expected, atol=1e-6)
    frame = Frame(entry, root)
    assert_allclose(frame.arrays["beauty"], denoised)
    assert_allclose(frame.arrays["target_beauty"], denoised * 2)
    inputs, targets, _ = batch([frame], [(0, 0, 0)], 32, 0, "cpu")
    assert_allclose(inputs["image"][0].numpy(), denoised)
    assert_allclose(targets["beauty"][0].numpy(), denoised * 2)
    features.extract(index, root, "depthpro2-s", device="cpu")
    assert len(observed) == 1
    with np.load(cache) as data:
        changed = {k: data[k] for k in data.files}
    changed["denoised"] *= 2
    save_arrays(cache, **changed)
    with pytest.raises(ValueError, match="input changed"):
        features.extract(index, root, "depthpro2-s", device="cpu")
    assert "image_sha256" in json.loads((root / "frame/context.json").read_text())

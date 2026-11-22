import hashlib
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import NoReturn

import numpy as np
import pytest
from PIL import Image

import strata_training.eval.real.fetch as acquisition
from strata_training.eval.real.fetch import download
from strata_training.eval.real.readers import decode_alpha, natural_samples


def test_download_publishes_only_verified_complete_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"eval-split-only"
    (source / "image.bin").write_bytes(payload)
    handler = partial(SimpleHTTPRequestHandler, directory=str(source))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "output.bin"
    url = f"http://127.0.0.1:{server.server_port}/image.bin"
    try:
        with pytest.raises(ValueError, match="SHA256"):
            download(url, destination, len(payload), "0" * 64)
        assert not destination.exists()
        with pytest.raises(ValueError, match="exceeded"):
            download(url, destination, len(payload) - 1)
        assert not destination.exists()
        report = download(url, destination, len(payload), hashlib.sha256(payload).hexdigest())
        assert destination.read_bytes() == payload
        assert report["bytes"] == 15
        destination.write_bytes(b"x" * len(payload))
        with pytest.raises(ValueError, match="Existing file SHA256"):
            download(url, destination, len(payload), hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_16bit_and_pairing_contract(tmp_path: Path) -> None:
    values = np.array([[0, 32768, 65535]], dtype=np.uint16)
    alpha = decode_alpha(Image.fromarray(values))
    assert alpha[0, 1] == 32768 / 65535
    assert alpha[0, 2] == 1
    (tmp_path / "image").mkdir()
    (tmp_path / "alpha").mkdir()
    Image.new("RGB", (3, 1), color="white").save(tmp_path / "image" / "id.jpg")
    Image.fromarray(values).save(tmp_path / "alpha" / "id.png")
    manifest = {
        "eval_only": True,
        "status": "downloaded",
        "expected": 1,
        "files": [{"path": "image/id.jpg"}, {"path": "alpha/id.png"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    sample = next(natural_samples(tmp_path))
    assert sample.identifier == "id"
    assert sample.image.shape == (1, 3, 3)
    assert sample.alpha[0, 1] == 32768 / 65535
    manifest["expected"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        next(natural_samples(tmp_path))


def test_total_and_dataset_budget_reject_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object) -> NoReturn:
        del args
        raise AssertionError("Budget allowed a transfer")

    monkeypatch.setattr(acquisition, "download", forbidden)
    monkeypatch.setitem(
        acquisition.CATALOG,
        "toy",
        {
            "source": "local-test",
            "licence": "test-only",
            "split": "test",
            "folders": {"image": "public"},
            "expected": 1,
        },
    )
    listing_size = 4

    def listing(folder: str) -> list[dict[str, str | int]]:
        del folder
        return [{"id": "one", "name": "one.png", "size": listing_size}]

    monkeypatch.setattr(acquisition, "drive_files", listing)
    occupied = tmp_path / "occupied"
    with occupied.open("wb") as file:
        file.truncate(80 * acquisition.GIB)
    result = acquisition.fetch(tmp_path, ["toy"])["toy"]
    assert result["status"] == "blocked"
    assert result["error"] == "Dataset/total byte budget exceeded"
    occupied.unlink()
    listing_size = 21 * acquisition.GIB
    result = acquisition.fetch(tmp_path, ["toy"])["toy"]
    assert result["error"] == "Dataset/total byte budget exceeded"

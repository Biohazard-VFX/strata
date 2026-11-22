import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from strata_training.eval.real.readers import samples
from strata_training.eval.real.runner import cache_path, cached, evaluate
from strata_training.io import save_arrays


def test_video_sampling_cached_layers_and_temporal_cadence(tmp_path: Path) -> None:
    root = tmp_path / "videomatte240k"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"eval_only": True, "status": "downloaded"}))
    with tarfile.open(root / "eval.tar", "w") as archive:
        for clip in ["a", "b"]:
            for frame in range(3):
                for channel in ["com", "pha"]:
                    value = frame * 127 if channel == "pha" else 64
                    image = Image.new("L" if channel == "pha" else "RGB", (2, 2), value)
                    data = io.BytesIO()
                    image.save(data, format="PNG")
                    info = tarfile.TarInfo(f"test/{clip}/{channel}/{frame:04d}.png")
                    info.size = len(data.getvalue())
                    archive.addfile(info, io.BytesIO(data.getvalue()))
    chosen = list(samples(root, frames_per_sequence=2))
    assert [(s.sequence, s.frame) for s in chosen] == [
        ("test/a/pha", 0),
        ("test/a/pha", 1),
        ("test/b/pha", 0),
        ("test/b/pha", 1),
    ]
    predictions = tmp_path / "predictions"
    for sample in chosen:
        save_arrays(
            cache_path(predictions, sample.identifier),
            alpha=np.stack([np.zeros((2, 2)), np.ones((2, 2))]),
        )
    output = tmp_path / "report.json"
    report = evaluate(root, output, cached(predictions), "cached", frames_per_sequence=2)
    assert report["evaluated"] == 4
    assert report["valid_images"]["dtssd"] == 2
    assert report["macro_mean"]["dtssd"] == 100 * 127 / 255
    assert report["macro_mean"]["full_mae"] == 127 / 510
    assert json.loads(output.read_text())["frames_per_sequence"] == 2

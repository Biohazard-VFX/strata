from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from strata_training.io import atomic_path, save_array, save_arrays, write_json


def test_atomic_publication_preserves_previous_file_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, {"step": 1})

    def interrupted_write() -> None:
        with atomic_path(path) as temporary:
            temporary.write_text("incomplete")
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_write()
    assert path.read_text() == '{\n  "step": 1\n}\n'
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="Out of range float"):
        write_json(path, {"step": float("nan")})
    assert path.read_text() == '{\n  "step": 1\n}\n'
    assert list(tmp_path.iterdir()) == [path]
    write_json(path, {"step": 2})
    assert '"step": 2' in path.read_text()


def test_array_publication_keeps_exact_names_and_values(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    single, multiple = tmp_path / "one.npy", tmp_path / "many.npz"
    save_array(single, values)
    save_arrays(multiple, alpha=values)
    assert_array_equal(np.load(single, allow_pickle=False), values)
    with np.load(multiple, allow_pickle=False) as data:
        assert data.files == ["alpha"]
        assert_array_equal(data["alpha"], values)
    assert set(tmp_path.iterdir()) == {single, multiple}

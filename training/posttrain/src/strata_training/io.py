import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from strata_training.types import Array


@contextmanager
def atomic_path(path: Path) -> Generator[Path, None, None]:
    """Own a temporary sibling until exit; publish on success; concurrent writers are last-wins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_array(path: Path, value: Array) -> None:
    with atomic_path(path) as temporary:
        np.save(temporary, value, allow_pickle=False)


def save_arrays(path: Path, **values: Any) -> None:
    with atomic_path(path) as temporary:
        np.savez_compressed(temporary, allow_pickle=False, **values)

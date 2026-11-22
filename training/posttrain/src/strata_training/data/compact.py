from pathlib import Path

import numpy as np

from strata_training.data.exr import exr
from strata_training.types import Layers


def read_layers(path: Path, layers: int = 4) -> Layers:
    """Return owned native compact targets; calls share no mutable state."""
    with exr.File(str(path), separate_channels=True) as file:
        channels = file.parts[0].channels
        inverse = np.stack([channels[f"L{k}.invdepth"].pixels for k in range(layers)])
        alpha = np.stack([channels[f"L{k}.A"].pixels for k in range(layers)])
        rgb = np.stack(
            [np.stack([channels[f"L{k}.{c}"].pixels for c in "RGB"]) for k in range(layers)]
        )
    return Layers(inverse.astype(np.float32), alpha.astype(np.float32), rgb.astype(np.float32))

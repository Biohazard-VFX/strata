from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]


@dataclass
class Layers:
    inverse_depth: Array
    alpha: Array
    rgb: Array

    @property
    def valid(self) -> Array:
        return (self.alpha > 1e-7) & (self.inverse_depth > 0)

    def arrays(self) -> dict[str, Array]:
        return {"inverse_depth": self.inverse_depth, "alpha": self.alpha, "rgb": self.rgb}


def fp16_safe(array: Array) -> Array:
    return array.astype(np.float16 if np.max(np.abs(array), initial=0) < 65000 else np.float32)

import numpy as np
from numpy.typing import NDArray

class KDTree:
    def __init__(self, data: NDArray[np.float64]) -> None: ...
    def query(
        self, x: NDArray[np.float64], k: int = 1
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]: ...

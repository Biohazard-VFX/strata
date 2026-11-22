import numpy as np
from numpy.typing import NDArray

class csr_matrix: ...

class coo_matrix:
    def __init__(
        self,
        arg: tuple[NDArray[np.float64], tuple[NDArray[np.int64], NDArray[np.int64]]],
        shape: tuple[int, int],
    ) -> None: ...
    def tocsr(self) -> csr_matrix: ...

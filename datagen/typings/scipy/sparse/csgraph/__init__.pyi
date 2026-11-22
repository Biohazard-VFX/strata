import numpy as np
from numpy.typing import NDArray

from scipy.sparse import csr_matrix

def dijkstra(csgraph: csr_matrix, *, directed: bool = True, indices: int) -> NDArray[np.float64]: ...
def connected_components(csgraph: csr_matrix, *, directed: bool = True) -> tuple[int, NDArray[np.int32]]: ...

from types import TracebackType
from typing import Self

import numpy as np
from numpy.typing import NDArray

class Channel:
    pixels: NDArray[np.generic]

class File:
    def __init__(self, filename: str, separate_channels: bool = False) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def channels(self) -> dict[str, Channel]: ...

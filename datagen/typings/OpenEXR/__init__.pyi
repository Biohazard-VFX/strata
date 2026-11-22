from collections.abc import Mapping
from types import TracebackType
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray

class Channel:
    pixels: NDArray[np.generic]
    def __init__(self, pixels: NDArray[np.generic]) -> None: ...

class Part:
    channels: dict[str, Channel]
    header: dict[str, object]
    def name(self) -> str: ...

ZIP_COMPRESSION: object
ZIPS_COMPRESSION: object
scanlineimage: object
deepscanline: object

class File:
    parts: list[Part]
    @overload
    def __init__(
        self, filename: str, separate_channels: bool = False, header_only: bool = False
    ) -> None: ...
    @overload
    def __init__(
        self, header: Mapping[str, object], channels: Mapping[str, NDArray[np.generic] | Channel]
    ) -> None: ...
    def write(self, filename: str) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

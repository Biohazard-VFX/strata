"""OpenEXR Core 3.5 scanline decoding into owned planar float32 arrays."""

import ctypes as c
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import OpenEXR
from numpy.typing import NDArray


class Channel(c.Structure):
    """Borrowed channel pointers, valid until their worker pipeline is destroyed; not shared for mutation."""

    _fields_ = [
        ("name", c.c_char_p),
        ("height", c.c_int32),
        ("width", c.c_int32),
        ("x_samples", c.c_int32),
        ("y_samples", c.c_int32),
        ("linear", c.c_uint8),
        ("bytes", c.c_int8),
        ("type", c.c_uint16),
        ("user_bytes", c.c_int16),
        ("user_type", c.c_uint16),
        ("pixel_stride", c.c_int32),
        ("line_stride", c.c_int32),
        ("ptr", c.c_void_p),
    ]


class Chunk(c.Structure):
    """Owned value copy of Core chunk metadata; immutable while decode workers borrow it."""

    _fields_ = [
        ("index", c.c_int32),
        ("x", c.c_int32),
        ("y", c.c_int32),
        ("height", c.c_int32),
        ("width", c.c_int32),
        ("level_x", c.c_uint8),
        ("level_y", c.c_uint8),
        ("type", c.c_uint8),
        ("compression", c.c_uint8),
        ("offset", c.c_uint64),
        ("packed", c.c_uint64),
        ("unpacked", c.c_uint64),
        ("sample_offset", c.c_uint64),
        ("sample_size", c.c_uint64),
    ]


class Pipeline(c.Structure):
    """One worker's mutable Core buffers, released by exr_decoding_destroy before its context closes."""

    _fields_ = [
        ("size", c.c_size_t),
        ("channels", c.POINTER(Channel)),
        ("channel_count", c.c_int16),
        ("flags", c.c_uint16),
        ("part", c.c_int),
        ("context", c.c_void_p),
        ("chunk", Chunk),
        ("skip", c.c_int32),
        ("ignore", c.c_int32),
        ("bytes", c.c_uint64),
        ("user", c.c_void_p),
        ("packed", c.c_void_p),
        ("packed_size", c.c_size_t),
        ("unpacked", c.c_void_p),
        ("unpacked_size", c.c_size_t),
        ("packed_counts", c.c_void_p),
        ("packed_counts_size", c.c_size_t),
        ("counts", c.POINTER(c.c_int32)),
        ("counts_size", c.c_size_t),
        ("scratch1", c.c_void_p),
        ("scratch1_size", c.c_size_t),
        ("scratch2", c.c_void_p),
        ("scratch2_size", c.c_size_t),
        ("alloc", c.c_void_p),
        ("free", c.c_void_p),
        ("read", c.c_void_p),
        ("decompress", c.c_void_p),
        ("realloc", c.c_void_p),
        ("unpack", c.c_void_p),
        ("quick", Channel * 5),
        ("counts_valid", c.c_int),
        ("compression_context", c.c_void_p),
        ("free_compression", c.c_void_p),
    ]


class Initializer(c.Structure):
    """Owned per-read Core settings; copied into the context, with a process-lifetime error callback."""

    _fields_ = [
        ("size", c.c_size_t),
        ("error_handler", c.c_void_p),
        ("alloc", c.c_void_p),
        ("free", c.c_void_p),
        ("user", c.c_void_p),
        ("read", c.c_void_p),
        ("query_size", c.c_void_p),
        ("write", c.c_void_p),
        ("destroy", c.c_void_p),
        ("max_width", c.c_int),
        ("max_height", c.c_int),
        ("max_tile_width", c.c_int),
        ("max_tile_height", c.c_int),
        ("zip_level", c.c_int),
        ("dwa_quality", c.c_float),
        ("flags", c.c_int),
        ("htj2k_quality", c.c_float),
        ("zstd_level", c.c_int),
    ]


@c.CFUNCTYPE(None, c.c_void_p, c.c_int, c.c_char_p)
def _quiet_error(_context: c.c_void_p, _code: int, _message: bytes) -> None:
    """Core reports failures through return codes, not unsolicited stderr output."""


def _library() -> c.CDLL:
    library = c.CDLL(OpenEXR.__file__)
    signatures = {
        "exr_start_read": [c.POINTER(c.c_void_p), c.c_char_p, c.c_void_p],
        "exr_finish": [c.POINTER(c.c_void_p)],
        "exr_read_scanline_chunk_info": [c.c_void_p, c.c_int, c.c_int, c.POINTER(Chunk)],
        "exr_decoding_initialize": [c.c_void_p, c.c_int, c.POINTER(Chunk), c.POINTER(Pipeline)],
        "exr_decoding_update": [c.c_void_p, c.c_int, c.POINTER(Chunk), c.POINTER(Pipeline)],
        "exr_decoding_choose_default_routines": [c.c_void_p, c.c_int, c.POINTER(Pipeline)],
        "exr_decoding_run": [c.c_void_p, c.c_int, c.POINTER(Pipeline)],
        "exr_decoding_destroy": [c.c_void_p, c.POINTER(Pipeline)],
    }
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = c.c_int
    library.exr_get_default_error_message.argtypes = [c.c_int]
    library.exr_get_default_error_message.restype = c.c_char_p
    return library


LIBRARY = _library()


def _check(code: int) -> None:
    if code:
        raise ValueError(f"OpenEXR Core: {LIBRARY.exr_get_default_error_message(code).decode()}")


def _counts(
    context: c.c_void_p, width: int, height: int, y_min: int
) -> tuple[NDArray[np.int64], list[Chunk], list[str]]:
    counts = np.empty(width * height, np.int64)
    chunks: list[Chunk] = []
    chunk = Chunk()
    _check(LIBRARY.exr_read_scanline_chunk_info(context, 0, y_min, c.byref(chunk)))
    pipeline = Pipeline()
    pipeline.size = c.sizeof(pipeline)
    try:
        _check(LIBRARY.exr_decoding_initialize(context, 0, c.byref(chunk), c.byref(pipeline)))
        pipeline.flags = 5
        _check(LIBRARY.exr_decoding_choose_default_routines(context, 0, c.byref(pipeline)))
        names = [pipeline.channels[i].name.decode() for i in range(pipeline.channel_count)]
        y = y_min
        while y < y_min + height:
            _check(LIBRARY.exr_read_scanline_chunk_info(context, 0, y, c.byref(chunk)))
            if chunk.width != width or chunk.height < 1 or chunk.y != y or y + chunk.height > y_min + height:
                raise ValueError("OpenEXR Core: invalid deep chunk dimensions")
            chunks.append(Chunk.from_buffer_copy(chunk))
            _check(LIBRARY.exr_decoding_update(context, 0, c.byref(chunk), c.byref(pipeline)))
            _check(LIBRARY.exr_decoding_run(context, 0, c.byref(pipeline)))
            size = chunk.width * chunk.height
            first = (y - y_min) * width
            counts[first : first + size] = np.ctypeslib.as_array(pipeline.counts, shape=(size + 1,))[:-1]
            y += chunk.height
        return counts, chunks, names
    finally:
        _check(LIBRARY.exr_decoding_destroy(context, c.byref(pipeline)))


def _decode(
    context: c.c_void_p,
    chunk: Chunk,
    arrays: dict[str, NDArray[np.float32]],
    offset: int,
    counts: NDArray[np.int64],
) -> None:
    @c.CFUNCTYPE(c.c_int, c.c_void_p)
    def validate(pointer: int) -> int:
        decoded = c.cast(pointer, c.POINTER(Pipeline)).contents
        found = np.ctypeslib.as_array(decoded.counts, shape=(counts.size + 1,))
        return 0 if int(found[-1]) == int(counts.sum()) and np.array_equal(found[:-1], counts) else 3

    pipeline = Pipeline()
    pipeline.size = c.sizeof(pipeline)
    try:
        _check(LIBRARY.exr_decoding_initialize(context, 0, c.byref(chunk), c.byref(pipeline)))
        pipeline.flags = 1
        pipeline.realloc = c.cast(validate, c.c_void_p).value
        for i in range(pipeline.channel_count):
            channel = pipeline.channels[i]
            channel.user_bytes = 4
            channel.user_type = 2
            channel.ptr = arrays[channel.name.decode()].ctypes.data + offset * 4
        _check(LIBRARY.exr_decoding_choose_default_routines(context, 0, c.byref(pipeline)))
        _check(LIBRARY.exr_decoding_run(context, 0, c.byref(pipeline)))
    finally:
        _check(LIBRARY.exr_decoding_destroy(context, c.byref(pipeline)))


def read(
    path: Path, width: int, height: int, y_min: int
) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float32]]]:
    """Owned counts and channels from part zero; independent calls and chunk workers are thread-safe."""
    context = c.c_void_p()
    initializer = Initializer(
        size=c.sizeof(Initializer),
        error_handler=c.cast(_quiet_error, c.c_void_p).value,
        zip_level=-2,
        dwa_quality=-1.0,
        zstd_level=-1,
    )
    _check(LIBRARY.exr_start_read(c.byref(context), os.fsencode(path), c.byref(initializer)))
    try:
        counts, chunks, names = _counts(context, width, height, y_min)
        if np.any(counts < 0):
            raise ValueError("OpenEXR Core: negative deep sample count")
        offsets = np.concatenate([[0], np.cumsum(counts)])
        arrays = {name: np.empty(int(offsets[-1]), np.float32) for name in names}
        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(
                    _decode,
                    context,
                    chunk,
                    arrays,
                    int(offsets[(chunk.y - y_min) * width]),
                    counts[(chunk.y - y_min) * width : (chunk.y - y_min + chunk.height) * width],
                )
                for chunk in chunks
            ]
            for future in futures:
                future.result()
        return counts, arrays
    finally:
        _check(LIBRARY.exr_finish(c.byref(context)))

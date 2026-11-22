"""Typed bridges between numpy arrays and USD value arrays, plus stage conventions."""

import os
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from strata_dataset import identity, roles

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def vec3f(values: NDArray[np.floating]) -> Vt.Vec3fArray:
    data = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 3)
    return cast("Vt.Vec3fArray", Vt.Vec3fArray.FromNumpy(data))  # pyright: ignore[reportUnknownMemberType]


def vec2f(values: NDArray[np.floating]) -> Vt.Vec2fArray:
    data = np.ascontiguousarray(values, dtype=np.float32).reshape(-1, 2)
    return cast("Vt.Vec2fArray", Vt.Vec2fArray.FromNumpy(data))  # pyright: ignore[reportUnknownMemberType]


def quath(values: NDArray[np.floating]) -> Vt.QuathArray:
    """Quaternions given as (w, x, y, z) rows."""
    rows = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    return Vt.QuathArray([Gf.Quath(float(w), float(x), float(y), float(z)) for w, x, y, z in rows])


def ints(values: NDArray[np.integer]) -> Vt.IntArray:
    data = np.ascontiguousarray(values, dtype=np.int32).ravel()
    return cast("Vt.IntArray", Vt.IntArray.FromNumpy(data))  # pyright: ignore[reportUnknownMemberType]


def floats(values: NDArray[np.floating]) -> Vt.FloatArray:
    data = np.ascontiguousarray(values, dtype=np.float32).ravel()
    return cast("Vt.FloatArray", Vt.FloatArray.FromNumpy(data))  # pyright: ignore[reportUnknownMemberType]


def matrix(rows: NDArray[np.floating]) -> Gf.Matrix4d:
    """A Gf matrix from a 4x4 array in USD's row-vector convention (translation in the last row)."""
    values = np.asarray(rows, dtype=np.float64).reshape(4, 4)
    return Gf.Matrix4d(*(tuple(float(v) for v in row) for row in values))


def new_stage(path: Path) -> Usd.Stage:
    """A Y-up, metre-unit stage whose time codes are frames at 24 fps."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetFramesPerSecond(24.0)
    return stage


def relative_asset(target: Path, layer_dir: Path) -> str:
    """An asset path relative to the authoring layer, so a build root can move as a whole.

    The path is lexical: symlinks inside the build root are kept, never followed out of it.
    """
    return Path(os.path.relpath(target, layer_dir)).as_posix()


def tag_instances(prim: Usd.Prim, ids: IntArray, name: str) -> None:
    """Set `instance_id` and its checks, a `coverage` of 1 and the role primvars of the id range `name`.

    Constant for one id, per instance otherwise. Karma premultiplies deep extra channels by the
    sample's float alpha but stores A as half, so `coverage` is the sample's float alpha. Each role
    primvar (`strata_dataset.roles.membership`) is 1 or 0; the sky's `foreground` 0 also feeds the
    flat geometry mask.
    """
    api = UsdGeom.PrimvarsAPI(prim)
    values = {
        "instance_check": identity.check(ids),
        "instance_check2": identity.check2(ids),
        "coverage": np.ones(ids.size, np.float32),
    }
    for role, member in zip(roles.ROLES, roles.membership(name), strict=True):
        values[role] = np.full(ids.size, member, np.float32)
    if ids.size == 1:
        api.CreatePrimvar("instance_id", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant).Set(int(ids[0]))
        for key, value in values.items():
            api.CreatePrimvar(key, Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(float(value[0]))
        return
    api.CreatePrimvar("instance_id", Sdf.ValueTypeNames.IntArray, UsdGeom.Tokens.vertex).Set(ints(ids))
    for key, value in values.items():
        array = cast("Vt.FloatArray", Vt.FloatArray.FromNumpy(value))  # pyright: ignore[reportUnknownMemberType]
        api.CreatePrimvar(key, Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(array)

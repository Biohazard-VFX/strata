"""Pre-render audit of a composed shot: every rendered surface carries the tags of its manifest range.

Karma renders an absent primvar as 0, so a surface missing its `instance_id` or a role primvar
renders as sky or as a non-hero without any error. Every renderable Gprim, through instance proxies,
must resolve an authored constant `instance_id` in a declared range and each role primvar equal to
`strata_dataset.roles.membership` of that range. PointInstancer prototypes take per-instance arrays
from their instancer, which must hold one id and one value per role for every instance; prototype
geometry must not carry tags of its own.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray
from pxr import Sdf, Usd, UsdGeom

from strata_dataset import roles

TAGS = ("instance_id", *roles.ROLES)
RENDERED = (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)


def _names(ranges: list[dict[str, object]]) -> dict[int, str]:
    out: dict[int, str] = {}
    for entry in ranges:
        first, count = cast("int", entry["first_id"]), cast("int", entry["count"])
        out |= dict.fromkeys(range(first, first + count), cast("str", entry["name"]))
    return out


def _value(primvar: UsdGeom.Primvar) -> NDArray[np.float64] | None:
    if not primvar or not primvar.HasAuthoredValue():
        return None
    value = primvar.Get()
    return None if value is None else np.atleast_1d(np.asarray(value, np.float64))


def _tag_errors(
    where: str, values: dict[str, NDArray[np.float64] | None], names: dict[int, str]
) -> list[str]:
    """Errors of one prim's resolved tags: per-instance arrays, or one value for a surface."""
    missing = [name for name, value in values.items() if value is None]
    if missing:
        return [f"{where}: no authored {', '.join(missing)}"]
    ids = cast("NDArray[np.float64]", values["instance_id"])
    undeclared = sorted({int(i) for i in ids if int(i) != i or int(i) not in names})
    if undeclared:
        return [f"{where}: instance ids {undeclared[:4]} outside the manifest ranges"]
    expected = np.array([roles.membership(names[int(i)]) for i in ids], np.float64)
    found = [cast("NDArray[np.float64]", values[role]) for role in roles.ROLES]
    if any(value.size != ids.size for value in found):
        return [f"{where}: role primvars do not have one value per instance"]
    wrong = [role for k, role in enumerate(roles.ROLES) if not np.array_equal(found[k], expected[:, k])]
    return [f"{where}: {', '.join(wrong)} differ from the roles of their id range"] if wrong else []


def _surface_errors(prim: Usd.Prim, names: dict[int, str]) -> list[str]:
    api = UsdGeom.PrimvarsAPI(prim)
    values = {name: _value(api.FindPrimvarWithInheritance(name)) for name in TAGS}
    return _tag_errors(str(prim.GetPath()), values, names)


def _instancer_errors(prim: Usd.Prim, names: dict[int, str]) -> list[str]:
    api = UsdGeom.PrimvarsAPI(prim)
    values = {name: _value(api.GetPrimvar(name)) for name in TAGS}
    errors = _tag_errors(str(prim.GetPath()), values, names)
    indices = UsdGeom.PointInstancer(prim).GetProtoIndicesAttr().Get()
    ids = values["instance_id"]
    if not errors and ids is not None and ids.size != len(indices or []):
        errors.append(f"{prim.GetPath()}: {ids.size} instance ids for {len(indices or [])} instances")
    return errors


def _prototype_errors(prim: Usd.Prim, root: Sdf.Path) -> list[str]:
    api = UsdGeom.PrimvarsAPI(prim)
    own: list[str] = []
    for name in TAGS:
        primvar = api.FindPrimvarWithInheritance(name)
        if _value(primvar) is not None and primvar.GetAttr().GetPrim().GetPath().HasPrefix(root):
            own.append(name)
    return [f"{prim.GetPath()}: prototype geometry overrides its instancer's {', '.join(own)}"] if own else []


def tag_errors(stage: Usd.Stage, ranges: list[dict[str, object]]) -> list[str]:
    """Why the composed `stage` would render tags its manifest `instance_ids` ranges do not declare."""
    names = _names(ranges)
    every = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    instancers = [prim for prim in every if UsdGeom.PointInstancer(prim)]
    prototypes = [
        root
        for prim in instancers
        for root in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetForwardedTargets()
    ]
    errors = [error for prim in instancers for error in _instancer_errors(prim, names)]
    for prim in every:
        if not UsdGeom.Gprim(prim) or UsdGeom.Imageable(prim).ComputePurpose() not in RENDERED:
            continue
        roots = [root for root in prototypes if prim.GetPath().HasPrefix(root)]
        errors += _prototype_errors(prim, roots[0]) if roots else _surface_errors(prim, names)
    return errors

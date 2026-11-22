"""Creature heroes over long shots: loop cyclic pose clips and vary the coat smoothly with time."""

from pathlib import Path
from typing import cast

import numpy as np
from pxr import Sdf, Usd

from strata_datagen.generators import morph
from strata_datagen.rng import stream

Json = dict[str, object]
COAT_LAYER = "hero_coat.usdc"


def animate(
    stage: Usd.Stage, hero_path: str, asset: Path, *, asset_path: str, frames: tuple[int, ...], seed: int
) -> Json:
    """Loop the pose's clip over the shot (handles included) and morph its coat; returns what was done.

    Keyed coat widths are large, so they go to a binary sublayer beside the shot layer.
    """
    layer = Sdf.Layer.FindOrOpen(str(asset))  # pyright: ignore[reportUnknownMemberType]
    record: Json = {}
    first, last = frames[0] - 1, frames[-1] + 1
    period = int(max(layer.ListTimeSamplesForPath("/hero/skin.points") or [0.0]))
    if period > 0:
        morph.author_loop(
            stage.GetPrimAtPath(hero_path), asset_path, "/hero", period=period, first=first, last=last
        )
        record["loop_frames"] = period
    groom = cast("Sdf.PrimSpec | None", layer.GetPrimAtPath("/hero/groom"))
    raw = None if groom is None else cast("Json | None", groom.customData.get("morph"))
    if groom is None or raw is None:
        return record
    widths = layer.GetAttributeAtPath("/hero/groom.widths").default
    counts = layer.GetAttributeAtPath("/hero/groom.curveVertexCounts").default
    base = np.asarray(widths, dtype=np.float64).reshape(len(counts), -1)
    length = cast("list[float]", raw["length"])
    density = cast("list[float]", raw["density"])
    params = morph.CoatMorph(
        morph.Range(length[0], length[1]),
        morph.Range(density[0], density[1]),
        float(cast("float", raw["min_period_s"])),
        int(cast("int", raw["key_step"])),
    )
    record["coat"] = morph.author_coat(
        stage,
        f"{hero_path}/groom",
        base,
        morph=params,
        frames=(first, last),
        rng=stream(seed, "coat", "morph"),
    )
    return record

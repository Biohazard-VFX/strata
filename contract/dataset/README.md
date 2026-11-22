# Strata dataset contract (v0.4.0)

The training side reads exactly this. Version string: `strata-dataset-0.4.0` (in every
`manifest.json` and `camera.json`, `strata_dataset.records.VERSION`). `schema.json` is the JSON
Schema for both files; `validate.py` checks a shot directory against it, against
`strata_dataset.records.errors` and against the image files. Datagen packing and raw QA refuse
records of any other version.

0.4.0 carries roles in linear deep and beauty channels (`foreground`, `hero`, `fur`): Karma sums
the channels of samples it merges, so role coverage stays exact through any merge, and role targets
and sidecar roles come from these channels, never from instance ids. Instance ids become a
best-effort per-object label: two per-id hashes (`instance_check`, `instance_check2`) and an
integer residual test reject merged samples, which get id -1; ids must lie in [0, 32768). Training
reads role coverage from `roles.<role>` targets or sidecar `roles.npy` and never derives roles from
ids. 0.3.0 shots have neither and do not satisfy it.

0.3.0 added role coverage to the targets of every frame (`roles.foreground`, `roles.hero`,
`roles.fur` in `layers/NNNN.exr`), declares `hero_groom` only when a groom is authored, and renders
deep without merging samples across depths. 0.2.0 shots lack role coverage on frames whose raw deep
was pruned and do not satisfy it.

## Layout

A dataset root holds one directory per shot; any directory with a `manifest.json` is a shot.

```
<shot_id>/
  manifest.json
  camera.json
  beauty/0001.exr ... 0024.exr
  deep/0001.exr ... 0024.exr     raw deep, kept on the frame stride (`manifest.render.deep_stride`)
  layers/0001.exr ... 0024.exr   K = 4 sky-inclusive layer targets and role coverage, every frame
  samples/<shot_id>_NNNN/        raw-sample sidecars of every recomposition frame (the deep stride)
  qa/NNNN.json      raw QA acceptance of each frame (identity of every input it validated)
  scene/            optional: the USD the shot was rendered from (reproducibility, not training input)
  qa.json           optional: the generator's acceptance report
```

Entries whose names start with a dot (`.lock`, `.staging/`) belong to the writer; readers ignore them.

Frames are numbered from 1 and zero-padded to four digits, at 24 fps. `manifest.render.frames` is the
rendered range: most shots are 8-frame clips, a minority are 24-96-frame sequences.

## Units and conventions

- Metres everywhere. World is Y-up.
- Camera space: x right, y up, the camera looks down -Z (OpenGL/USD). `world_to_camera` is a
  4x4 row-major matrix for column vectors: `p_cam = M @ [x, y, z, 1]`.
- Pixel coordinates: origin at the top-left corner of the top-left pixel, u right, v down. A
  camera-space point projects to `u = cx + fx * x / -z`, `v = cy - fy * y / -z`.
- Z (flat and deep) is camera-plane depth: the distance along the optical axis, `-z_cam`,
  positive in front of the camera. Sky pixels hit the 3000 m backdrop (instance id 0).
- Colour: scene-linear ACEScg, not display-referred. Display and log transforms are applied
  at training time.
- Lens distortion: none in v0 (`distortion.model = "none"`).

## camera.json

`frames` has one record per rendered frame: `frame`, `width`, `height`, intrinsics `fx fy cx cy`
in pixels, `world_to_camera`, `distortion`, `shutter_open`/`shutter_close` in frames relative to
the frame (both 0 when motion blur is off), `fstop` (0 means depth of field off),
`focus_distance_m`, `focal_length_mm`, `sensor_width_mm`. `move` names the camera move
(`locked`, `pan_tilt`, `handheld`, `dolly`, `orbit`).

## manifest.json

`shot_id`, `seed`, `recipe_version`, `generator_versions` (module to version), `spec` (the fixed
choices the shot was built from), `scene` (sampled terrain, lighting, hero, camera summary),
`instance_ids` (named id ranges: `first_id`, `count`, and for scatter layers the prototype keys),
`assets` (every asset used: `id`, `source`, `license`, `origin`), `render` (renderer, resolution,
samples, frame range, AOV lists) and `timings` (filled after rendering: device, per-frame seconds,
VRAM).

## beauty/NNNN.exr

Multi-part scanline EXR, one part per AOV, all at the camera resolution:

| part | channels | meaning |
| --- | --- | --- |
| `C` | `R G B A` | raw path-traced beauty (half); the deep flattens to this |
| `albedo` | `albedo.R/G/B` | diffuse albedo |
| `N` | `N.x/y/z` | shading normal in camera space (half) |
| `roughness` | `roughness.roughness` | specular roughness of the front surface |
| `metallic` | `metallic.metallic` | metalness of the front surface |
| `Z` | `Z.Z` | camera-plane depth of the nearest surface (float, min filter) |
| `instance_id` | `instance_id.instance_id` | id of the nearest surface (float holding an integer) |
| `instance_check` | `instance_check.instance_check` | `check(id)` of the nearest surface (float in [1, 2)) |
| `instance_check2` | `instance_check2.instance_check2` | `check2(id)` of the nearest surface (float in [1, 2)) |
| `denoised` | `denoised.R/G/B/A` | `C` denoised by OIDN with albedo and normal guides: the model input; targets stay raw |
| `foreground` | `foreground.foreground` | geometry coverage of the pixel (float; 0 where only the sky is seen) |
| `hero` | `hero.hero` | hero coverage of the pixel (float), the flat image of the `hero` role primvar |
| `fur` | `fur.fur` | fur, feather and hair coverage of the pixel (float), the `fur` role primvar |
| `coverage` | `coverage.coverage` | flat copy of the deep sample-alpha channel: 1 wherever any surface, sky included, is seen; not a foreground mask |

## layers/NNNN.exr

Scanline EXR with, for each layer k in 0..3 (front to back), float32 `L<k>.invdepth` (1/metres,
visible-coverage-weighted), `L<k>.A` (the layer's conditional alpha) and `L<k>.R/G/B`
(premultiplied colour composited within the layer). They are `strata_dataset.layers.reduce_samples`
applied to every deep sample of the frame, sky included, with `sky` = (`foreground` below half the
sample's alpha, `strata_dataset.roles.sky`): samples of visible
coverage at most 1e-12 are dropped, the rest split at the largest three log-depth gaps above
ln(1.15) and merge front to back. Any layer holding visible sky keeps its alpha and colour and has
inverse depth 0. Sample alpha is the deep `coverage` channel (float), not the half `A`. Compositing
the four layers front to back reproduces the beauty `C`; an opaque sky-only pixel is one layer with
alpha 1, the sky colour and inverse depth 0. Training reads these for every frame.

The same file holds, for every frame, float32 `roles.<role>` (`strata_dataset.roles.coverage`): the
visible coverage of the role in the pixel, the sum over the frame's deep samples of the sample's
role channel (premultiplied, nested by `strata_dataset.roles.nested`) times the transmittance in
front, from the frame's own deep before any raw deep is pruned. A merged sample contributes exactly
its sources' share of each role; instance ids play no part. Every surface's role primvars follow its
`manifest.instance_ids` range name (`strata_dataset.roles.membership`):

| role | id range names |
| --- | --- |
| `foreground` | every range but `sky` |
| `hero` | `hero_skin`, `hero_groom`, `hero_hair` |
| `fur` | `hero_groom`, `hero_hair` (strands: fur, feathers, hair) |

`0 <= fur <= hero <= foreground <= 1` in every pixel; the sky's coverage is the pixel coverage minus
`foreground`.

## samples/<shot_id>_NNNN/

The raw samples of a recomposition frame (every frame on the deep stride from the first frame),
as uncompressed C-order NPY (`strata_dataset.samples`). A frame directory is write-once: it appears
whole, by one rename, and is never changed afterwards; only an offline removal deletes it. It is a
real directory, never a symlink. `identity.json` records the SHA-256 of every file below, and its
own SHA-256 is the frame's identity (recorded by raw QA). `strata_dataset.samples.read` opens the
directory once, reads every file through that handle and verifies each against `identity.json`:

| file | dtype, shape | meaning |
| --- | --- | --- |
| `noisy.npy` | float32 (3, H, W) | beauty `C` RGB |
| `denoised.npy` | float32 (3, H, W) | beauty `denoised` RGB |
| `deep/shape.npy` | int64 (2,) | (H, W) |
| `deep/offsets.npy` | int64 (H*W+1,) | row-major pixel sample offsets, first 0, last N |
| `deep/z.npy` | float32 (N,) | camera-plane depth, front to back within a pixel (stable for ties) |
| `deep/alpha.npy` | float32 (N,) | the sample's conditional alpha: deep `coverage` clipped to [0, 1] |
| `deep/rgb.npy` | float32 (N, 3) | premultiplied colour |
| `deep/roles.npy` | float32 (N, 3) | premultiplied `foreground`, `hero`, `fur` coverage of the sample: the deep role channels with 0 <= fur <= hero <= foreground <= alpha (`strata_dataset.roles.nested`) |
| `deep/ids.npy` | int64 (N,) | best-effort instance id (`strata_dataset.identity.decode`), -1 where it fails; for per-object use only, never for roles |
| `identity.json` | JSON | `{"sha256": {"<file>": "<hex>"}}` for each file above |

## deep/NNNN.exr

Single-part deep scanline EXR with channels `R G B A` (half), `Z ZBack instance_id instance_check
instance_check2 coverage foreground hero fur` (float), one sample list per pixel, samples unsorted.
`R G B` are premultiplied by `A`. Every other channel but `Z ZBack` is a constant per-surface primvar
premultiplied by the sample's alpha at float precision: `coverage` is 1 on every surface (sky
included), so it is the sample's float alpha; do not divide by the half `A`, which is off by up to
id x 2^-11 and rounds alpha above 0.99976 to 1. The role channels are 1 or 0 per surface. Karma
merges samples at equal depth (deep compression 5, depth bias 0) by summing their channels, so a
merged sample holds exactly its sources' role coverage but the coverage-weighted mean of their ids
and checks, which can round to an absent id (an equal merge of ids 1 and 3 reads as 2) or to one
holding only part of the sample.

`strata_dataset.identity.decode` gives each sample's best-effort id: id = round(instance_id /
coverage), kept only when id is in [0, 32768), |instance_id / coverage - id| <= max(0.05, 2e-5 x id)
and both |instance_check / coverage - check(id)| and |instance_check2 / coverage - check2(id)| are
at most 2^-14 (`identity-check-2`). `check(id)` is 1 + 2^-23 x the top 23 bits of the murmur3 32-bit
finalizer of (id + 0x9E3779B9) mod 2^32; `check2(id)` applies the finalizer again to that 32-bit hash
plus 0x7F4A7C15. Both are exact float32 in [1, 2); unmerged samples meet them within about 1e-5.
Bound: if a merged sample's sources other than id hold share m, each check misses check(id) by m
times a difference of independent uniform hashes, so it passes with probability at most
p = 2 x 2^-14 / m + 2^-23 and both with at most p^2 (6e-8 at m = 1/2, 1.5e-6 at m = 1/10), whatever
the number of sources, their shares and whether id is among them. The expected mislabelled share
accepted per merged sample is at most 2^-13. Raw QA bounds the fraction of samples and of coverage
with id -1.

ZBack - Z stays at most about 1 mm for surfaces.
Fully occluded surfaces are not stored. Beauty and deep share a one-pixel box filter: compositing
the samples front to back reproduces the beauty `C` part.

Every pixel ends in a sky sample: a camera-only backdrop sphere (radius 3000 m, the HDRI mapped as
Karma maps its dome, same exposure) with instance id 0 and every role channel 0, so
compositing all samples front to back reproduces the beauty `C` everywhere, partly covered pixels
over sky included. Layer targets and sidecars keep sky samples. The background sample is opaque,
finite (camera-plane Z up to 3000 m, below the 5000 m far clip; never inf) with Z == ZBack, and
`manifest.background` records its reserved id (0) and radius: compositors drop it with one
DeepExpression on instance_id; training finds it by its zero `foreground` role, masks depth losses
on it (sky depth is undefined) and supervises its colour and coverage.

Instance ids: 0 is the sky backdrop, 1 terrain, 2 hero skin, 3 hero groom (hair or fur grown on the
skin), 4 hero hairstyle, 100 and up thin structures, 1000 and up scattered instances, all below 32768
(`identity.MAX_ID`; generation refuses a scene that needs more).
`manifest.instance_ids` maps every id to its layer and declares only ranges the scene authors: a hero
without a groom has no `hero_groom`, so an id 3 in its deep is invalid.

# Strata dataset contract (v0.4.0)

The training side reads exactly this. Version string: `strata-dataset-0.4.0` (in every
`manifest.json` and `camera.json`, `strata_dataset.records.VERSION`). `schema.json` is the JSON
Schema for both files; `validate.py` checks a shot directory against it, against
`strata_dataset.records.errors` and against the image files. Datagen packing and raw QA refuse
records of any other version.

0.4.0 adds the deep and beauty `instance_check` channel: a per-id hash that rejects samples Karma
merged from several surfaces, whose averaged id can round to a declared id of a surface that is not
there. Such samples get id -1 in the sidecars and count for no role. Ids must lie in [0, 32768).
0.3.0 shots have no identity check and do not satisfy it.

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
| `denoised` | `denoised.R/G/B/A` | `C` denoised by OIDN with albedo and normal guides: the model input; targets stay raw |
| `foreground` | `foreground.foreground` | geometry coverage of the pixel (float; 0 where only the sky is seen) |
| `coverage` | `coverage.coverage` | flat copy of the deep sample-alpha channel: 1 wherever any surface, sky included, is seen; not a foreground mask |

## layers/NNNN.exr

Scanline EXR with, for each layer k in 0..3 (front to back), float32 `L<k>.invdepth` (1/metres,
visible-coverage-weighted), `L<k>.A` (the layer's conditional alpha) and `L<k>.R/G/B`
(premultiplied colour composited within the layer). They are `strata_dataset.layers.reduce_samples`
applied to every deep sample of the frame, sky included, with `sky` = (id == 0): samples of visible
coverage at most 1e-12 are dropped, the rest split at the largest three log-depth gaps above
ln(1.15) and merge front to back. Any layer holding visible sky keeps its alpha and colour and has
inverse depth 0. Sample alpha is the deep `coverage` channel (float), not the half `A`. Compositing
the four layers front to back reproduces the beauty `C`; an opaque sky-only pixel is one layer with
alpha 1, the sky colour and inverse depth 0. Training reads these for every frame.

The same file holds, for every frame, float32 `roles.<role>` (`strata_dataset.roles`): the visible
coverage of the role's samples in the pixel, the sum over its samples of alpha times the
transmittance in front, from the frame's own deep before any raw deep is pruned. Samples that fail
the identity check still occlude the samples behind them but belong to no role, so a pixel's roles
can sum to less than its coverage; raw QA bounds that loss (`unlabelled_coverage`). A role names
`manifest.instance_ids` ranges, never fixed ids:

| role | id ranges |
| --- | --- |
| `foreground` | every range but `sky` |
| `hero` | `hero_skin`, `hero_groom`, `hero_hair` |
| `fur` | `hero_groom`, `hero_hair` (strands: fur, feathers, hair) |

A role with no declared range has zero coverage. `fur <= hero <= foreground` in every pixel.

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
| `deep/ids.npy` | int64 (N,) | `strata_dataset.identity.decode`: the id, or -1 where the identity check fails; sky samples keep id 0 |
| `identity.json` | JSON | `{"sha256": {"<file>": "<hex>"}}` for each file above |

## deep/NNNN.exr

Single-part deep scanline EXR with channels `R G B A` (half), `Z ZBack instance_id instance_check
coverage` (float), one sample list per pixel, samples unsorted. `R G B` are premultiplied by `A`.
`instance_id`, `instance_check` and `coverage` (a constant 1 on every surface, sky included) are
premultiplied by the sample's alpha at float precision, so `coverage` is the sample's float alpha;
do not divide by the half `A`, which is off by up to id x 2^-11 and rounds alpha above 0.99976 to 1.
Karma merges samples at equal depth (deep compression 5, depth bias 0) by summing their channels,
so a merged sample of two surfaces holds the coverage-weighted mean of their ids, which can round
to a third, declared id (an equal merge of ids 1 and 3 reads as 2).

`strata_dataset.identity.decode` gives each sample's id: id = round(instance_id / coverage), kept
only when id is in [0, 32768) and |instance_check / coverage - check(id)| <= 2^-14. `check(id)`
(`identity-check-1`) is 1 + 2^-23 x the top 23 bits of the murmur3 32-bit finalizer of
(id + 0x9E3779B9) mod 2^32, an exact float32 in [1, 2); unmerged samples meet it within about 1e-5.
A merge of different ids passes with probability at most about 4 x 2^-14 (2.4e-4) per merged
sample, or when the other surfaces' share is below 2^-14 / |check difference|. Raw QA bounds the
fraction of failing samples and the coverage they hold.

ZBack - Z stays at most about 1 mm for surfaces.
Fully occluded surfaces are not stored. Beauty and deep share a one-pixel box filter: compositing
the samples front to back reproduces the beauty `C` part.

Every pixel ends in a sky sample: a camera-only backdrop sphere (radius 3000 m, the HDRI mapped as
Karma maps its dome, same exposure) with instance id 0, so compositing all samples front to back
reproduces the beauty `C` everywhere, partly covered pixels over sky included. Layer targets and
sidecars keep id-0 samples. The background sample is opaque,
finite (camera-plane Z up to 3000 m, below the 5000 m far clip; never inf) with Z == ZBack, and
`manifest.background` records its reserved id (0) and radius: compositors drop it with one
DeepExpression on instance_id; training masks depth losses on it (sky depth is undefined) and
supervises its colour and coverage.

Instance ids: 0 is the sky backdrop, 1 terrain, 2 hero skin, 3 hero groom (hair or fur grown on the
skin), 4 hero hairstyle, 100 and up thin structures, 1000 and up scattered instances, all below 32768
(`identity.MAX_ID`; generation refuses a scene that needs more).
`manifest.instance_ids` maps every id to its layer and declares only ranges the scene authors: a hero
without a groom has no `hero_groom`, so an id 3 in its deep is invalid.

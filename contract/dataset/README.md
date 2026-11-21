# Strata dataset contract (v0.1.0)

The training side reads exactly this. Version string: `strata-dataset-0.1.0` (in every
`manifest.json` and `camera.json`). `schema.json` is the JSON Schema for both files;
`validate.py` checks a shot directory against it and against the image files.

## Layout

A dataset root holds one directory per shot; any directory with a `manifest.json` is a shot.

```
<shot_id>/
  manifest.json
  camera.json
  beauty/0001.exr ... 0024.exr
  deep/0001.exr ... 0024.exr
  scene/            optional: the USD the shot was rendered from (reproducibility, not training input)
  qa.json           optional: the generator's acceptance report
```

Frames are numbered from 1 and zero-padded to four digits. v0 shots have 24 frames at 24 fps.

## Units and conventions

- Metres everywhere. World is Y-up.
- Camera space: x right, y up, the camera looks down -Z (OpenGL/USD). `world_to_camera` is a
  4x4 row-major matrix for column vectors: `p_cam = M @ [x, y, z, 1]`.
- Pixel coordinates: origin at the top-left corner of the top-left pixel, u right, v down. A
  camera-space point projects to `u = cx + fx * x / -z`, `v = cy - fy * y / -z`.
- Z (flat and deep) is camera-plane depth: the distance along the optical axis, `-z_cam`,
  positive in front of the camera. Pixels that see only the sky have flat `Z = 0` and no deep samples.
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
| `C` | `R G B A` | beauty (half) |
| `albedo` | `albedo.R/G/B` | diffuse albedo |
| `N` | `N.x/y/z` | shading normal in camera space (float) |
| `roughness` | `roughness.roughness` | specular roughness of the front surface |
| `metallic` | `metallic.metallic` | metalness of the front surface |
| `diffuse_direct`, `diffuse_indirect` | `<part>.R/G/B` | diffuse lighting split (LPE `C<RD>L`, `C<RD>.+L`) |
| `specular_direct`, `specular_indirect` | `<part>.R/G/B` | glossy/specular reflection split |
| `Z` | `Z.Z` | camera-plane depth of the nearest surface (float, min filter) |
| `instance_id` | `instance_id.instance_id` | id of the nearest surface (float holding an integer) |

## deep/NNNN.exr

Single-part deep scanline EXR with channels `R G B A Z ZBack instance_id`, float, one sample list
per pixel, samples unsorted. `R G B` are premultiplied by `A`. `instance_id` is premultiplied by
`A` as well: the integer id of a sample is `round(instance_id / A)`. Karma merges samples of the
same surface that lie within 1 mm in depth (deep compression 5), so a sample can stand for several
pixel samples; ZBack - Z stays at most about 1 mm for surfaces. Fully occluded surfaces are not
stored. Beauty and deep share a one-pixel box filter: compositing the samples front to back
reproduces the beauty `C` part (mean absolute difference about 4e-5).

Instance ids: 0 is sky (no deep sample), 1 terrain, 2 hero skin, 3 hero groom (hair or fur),
1000 and up are scattered instances; `manifest.instance_ids` maps every id to its layer.

# Datagen decisions

- USD is authored with usd-core in a uv project, not Houdini's pxr, so datagen is typed, testable without a license and runs unchanged on render nodes.
- Materials are MaterialX standard_surface UsdShade networks on `outputs:mtlx:surface`; node ids and inputs are validated against the MaterialX library.
- Roughness and metallic AOVs are `inputs:aov:<name>` on the surface shader, rendered as raw render vars.
- Camera-space normals come from Karma's `ray:hitN_camera`: x right, y up, +Z towards the viewer.
- Deep carries `C` and `instance_id`; ids are alpha-premultiplied like colour, so a sample's id is `round(id / A)`.
- Deep uses Karma's default compression (level 5); uncompressed deep is too large for datasets at 1280x720.
- Beauty, AOVs and deep share a one-pixel box filter so the flattened deep reproduces the beauty.
- Beauty EXRs are multi-part, one part per AOV, ZIPS-compressed.
- Every product name is time-sampled per frame; outputs never rely on `$F` expansion.
- Camera focal length and apertures are authored in tenths of a scene unit (mm / 100 at metres), per UsdGeomCamera.
- Dome lights set `inputs:karma:light:renderlightgeo` so the HDRI is visible to camera rays.
- HDRI directions use Karma's latlong frame: image centre at +Z, top row at +Y.
- Each HDRI's sun is cut into a distant light whose intensity is the measured irradiance (Karma's distant-light unit).
- Shots are normalised to a fixed horizontal irradiance (sky plus sun) so exposure is comparable across HDRIs.
- Camera-visible skies use 8K outdoor HDRIs resampled to 4K.
- Path-traced samples default to 64 per pixel.
- FBX models are converted in Blender, then merged into one Y-up metre mesh per variation, footprint-centred on y = 0.
- Megascans maps are stored at most 2K, 8-bit: sRGB albedo, raw normal, roughness and opacity; EXR albedo is linear.
- Ground surfaces are natural Megascans surfaces only; asphalt, concrete, metal and wood are excluded.
- Heroes enter only through an external library (`hero.json` + USD poses); repo code never names a hero.
- Fur and hair come from the generic groom generator: guide shapes blended from the nearest guides, or parametric hair when there are none.
- Grooms and skins are baked per pose into the build root at prepare time, so shot layers stay small.
- Presets are the only code that names library asset ids; generators take assets as parameters.
- Banned identifiers are checked as SHA-256 digests so the public repo never spells them.
- Randomness comes from named streams `sha256(seed:labels)`; fixed choices still consume their draws, so a seed gives byte-identical USD.
- Terrain is a 300 m, 0.5 m-cell heightfield with a flattened centre; beyond the scattered 60 m it flattens and sinks, so bare ground never rises into the HDRI horizon.
- Renders are Karma XPU GPU-only and abort without a GPU; the driver retries once with the CPU device and records which ran.
- Job lists (`jobs.json`, `jobs.sh`) hold one husk command per shot with paths relative to the build root.
- Files are written atomically (temporary file, then rename).
- Bulk shots are long continuous 720-frame shots; diversity comes from smooth seeded B-spline curves over time (timeline.Curve), never cuts.
- Every frame gets K=4 layer targets (contract strata_contract.layers, the training merge rule); raw deep is kept every 24th frame, as long shots are 720 frames.


- Bulk shots are 8-frame clips for diversity; a configurable share (default 10%) are 24-96-frame sequences for temporal training. Camera moves scale with duration.
- Human hair comes from the hairstyle generator: guides grown under gravity against the skin, dense strands interpolated, then Blender hair-node deformers ported to numpy (strands.py).
- Hair renders with Karma's kma_hair melanin shader on `outputs:kma:surface`; Karma XPU skips MaterialX chiang_hair graphs (ND_surface roots).
- Heroes mark hair-growing skin with float vertex primvars `region_<scalp|brows|lashes|beard>`; a scalp region gives the hero a seeded hairstyle in every shot.
- Hairstyles are grown per shot into `scene/hair.usdc` so each seed gets its own style; skins and fur grooms stay baked per pose at prepare time.

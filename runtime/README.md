# Strata P0 runtime

C11 `libstrata` chains frozen single-frame Depth Pro 2 H+ and a four-layer ONNX head. The CLI writes metric DP2 depth and plate-coloured layered deep. Smoke checkpoints are **plumbing-only**, not trained models or production candidates.

## Build

Run commands from the repository root. Install a C11 compiler, CMake >=3.20, Ninja, pkg-config, OpenEXR 3 development files, json-c and OpenSSL development files, and clang-format 23.1.4. `stb_image` is vendored with its license. CUDA builds also need the CUDA/cuDNN shared libraries required by the selected ONNX Runtime SDK.

```sh
uv sync --project training --extra cuda --locked --python 3.12
env -u PYTHONPATH training/.venv/bin/python runtime/cmake/fetch_ort.py
cmake --preset normal -S runtime \
  -DSTRATA_TEST_PYTHON="$PWD/training/.venv/bin/python"
cmake --build build/normal --parallel 4
build/normal/strata --help
```

The SDK fetcher pins ONNX Runtime 1.32.0, records download sources/checksums, and refuses a nonempty destination. It uses official release artifacts only: Microsoft's release archive, else the official NuGet headers and PyPI native libraries. To use an existing SDK, pass `-DONNXRUNTIME_ROOT=/absolute/sdk` with `include/onnxruntime_c_api.h`, `lib/libonnxruntime.so` and, for CUDA, `lib/libonnxruntime_providers_cuda.so` beside it. Binaries, weights and build products are never committed.

For a CPU-only development environment use `--extra cpu`, fetch with `--cpu`, and run `tools/check_all.sh --cpu`. This mode runs the unsanitized `cpu` preset with GPU tests excluded. CPU contract tests use tiny ONNX fixtures (depth, focal and a contract-violating model); they do not run the full H+ model. `tools/install_hooks.sh` installs the Conventional Commit message hook.

## Export

Export needs a CUDA GPU and the pinned Apple source installed by `uv sync`. The first run downloads the original H+ and focal checkpoints into the Torch cache. Treat the Apple research-checkpoint license as a separate commercial-release gate.

```sh
gpu_run() {
  systemd-run --user --wait --pipe --collect -q --working-directory="$PWD" \
    -p MemoryMax=32G -p MemorySwapMax=0 timeout 3600 env -u PYTHONPATH "$@"
}
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.export \
  --precision fp16 --out models/runtime-p0/depthpro2-h.onnx
```

Keep all five generated files together: `depthpro2-h.onnx` and `.onnx.weights`, `depthpro2-h.fov.onnx` and its `.weights`, and `depthpro2-h.json`, the exact version-two contract manifest. The exporter finishes with `strata_train.optimize`, which applies ONNX Runtime's basic graph transforms once, rewrites stride==kernel ConvTranspose as a 1x1 Conv plus DepthToSpace and fuses each float32 rotate-half RoPE into one `com.microsoft.RotaryEmbedding`; run `python -m strata_train.optimize model.onnx model.fov.onnx` on a CUDA host to bring an older export up to date. The runtime rejects a missing or incompatible manifest at open and after unload, validates ONNX tensor names/types/shapes, and checks the actual output shape at inference. The manifest describes semantics; it is **not** a cryptographic binding to the weights.

### Layered head

Export a current `coverage-simplex-bce-v1` checkpoint without retraining or re-exporting DP2:

```sh
uv sync --project training/posttrain --extra cpu --locked
training/posttrain/.venv/bin/python -m strata_training.model.export \
  --checkpoint /path/to/head.pt --backbone models/runtime-p0/depthpro2-h.onnx \
  --out models/runtime-p0/head.onnx
```

The existing DP2 graph and weights are not edited. Its public output contains only inverse depth, not the 128 feature channels the head needs. Export therefore writes `head.backbone.onnx`: identical DP2 nodes and weights plus one float32 feature output. C runs that sidecar **once**, then the separate `head.onnx`. Head metadata binds both graph files by SHA-256. External weights are reused with hard links or copied alongside the sidecar. The sidecar is required; there is no reconstruction of missing features from depth, duplicate encoder pass or old-objective compatibility mode.

The head runs float32 in 256-pixel cores with a 32-pixel halo, matching its receptive field. It receives cropped frozen features, ACEScg-converted model RGB and camera-converted inverse metres. Export ignores learned sample colour: this plumbing uses the approved plate-only DeepRecolor-style colour policy. Depth-only inference and position output remain DP2 outputs. `strata_run` uses the original backbone without feature outputs even when a head is configured; switching between depth and deep reloads the backbone session. Deep inference still retains the full-frame feature output on host and device; only head evaluation is tiled.

Deep feature buffers scale with frame size: about 4 GB host + 4 GB GPU at UHD and about 17 GB each at 7680×4352; full-frame deep above UHD needs streaming (follow-up).

For deterministic, untrained plumbing qualification only, add `--create-plumbing-only`, with both checkpoint and ONNX filenames containing `plumbing-only`. It refuses to overwrite an existing checkpoint; checkpoint metadata says `purpose=plumbing-only`, `trained=false`. Never publish or commit those artifacts. Model contract v2 removes the retired single-sample deep contract: regenerate only the JSON manifest beside an existing DP2 export using `strata_contract.write_manifest`; no DP2 graph or weights need changing.

## CLI

```sh
gpu_run build/normal/strata depth \
  --model models/runtime-p0/depthpro2-h.onnx --head models/runtime-p0/head.onnx \
  --in plate.exr --in-colorspace lin-rec709 --f-px 2666.67 --provider cuda \
  --out-depth depth.exr --out-deep deep.exr --out-position P.exr
```

Depth output is required. Deep output requires `--head`; omit both for depth-only inference. An output whose resolved parent lies inside either model's directory (the real one too, if the model is a symlink), or that resolves to an input, the model, its manifest, its `.fov.onnx` or another output (through `.`, `..`, symlinks or hard links), is a usage error. ONNX Runtime only reads external data inside the model's directory, so model files are never parsed to find it. Without `--f-px` or camera intrinsics, focal is estimated by the sibling `.fov.onnx`; a field-of-view output that is not finite and strictly between 0 and 180 degrees fails the run. `--exposure stops` affects scene-linear input's inference conversion, not the preserved deep RGB. `--device N` selects the CUDA device. The CLI prints one JSON object: `provider`, `fallback_reason`, the resolved `focal_px`/`focal_source`, `nonfinite_input_pixels`, `session_ms`, `frame_ms`, `write_ms` and per-stage `stages` (prepare, focal, upload, run, download, postprocess). File reads and writes are outside `frame_ms`.

`--in-colorspace` is required, with no default, metadata inference or filename guessing: `acescg` (AP1/D60), `aces2065-1` (AP0/D60), `lin-rec709` (sRGB primaries/D65) or `lin-rec2020` (Rec.2020/D65). Only linear values are accepted; decode log/camera/display encodings externally. Unsupported names, including `LogC`, `S-Log3` and `srgb`, are usage errors listing the supported spaces. Inputs are already-oriented 8-bit PNG/JPEG or scanline RGB EXR; the declaration describes the numeric RGB values regardless of file type. Tiled, deep and subsampled EXRs are rejected. Exact 3x3 matrices convert only model input to the training beauty space, ACEScg; D65 sources use the OCIO ACES 1.3 studio configuration's Bradford adaptation to D60. Matrix arithmetic uses doubles and rounds once to float32, with an identity bypass for ACEScg. Encoder preprocessing then matches training's ACEScg display matrix before x=max(0,0.72*2^stops*C); Fable shoulder is x below 0.6, otherwise 0.6+0.4*(x-0.6)/(0.4+x-0.6), then sRGB and ImageNet normalization. Negative/NaN/-inf map to zero; +inf maps to one. Input remains float32, without 8-bit quantization. Exposure is one fixed value per shot; --exposure adds an override in stops, never adapts per frame. The frozen encoder runs once. Inference runs at native resolution with bottom/right replicate padding to 32 and crops back. The contract allows padded height <=4352 and width <=7680, subject to device memory.

The flat EXR has float32 `Z` and `network_inverse_depth`, ZIP-compressed. For original width `W`, focal `f_px` and network output `q`, `Z = f_px / (W * q)`: optical-axis camera depth in metres, not radial range; zero `q` is `Z=+inf`. Deep EXR has `R G B A Z ZBack` as uncompressed float32 samples, front-to-back, with one sample for each nonempty predicted layer (at most four). Zero-visible-coverage layers are omitted; equal depths are not merged. `A` is conditional coverage and each sample RGB is plate RGB times that alpha, so front-to-back compositing reproduces the plate. `ZBack=Z`. Deep RGB stays in the declared plate space, not ACEScg; the EXR header records its `chromaticities` and `strata:color_space` name. No colour unmixing is performed. Every writer renders into a sibling temporary file and renames it over the target.

### Camera and position

Use `--camera camera.json` instead of `--f-px` for calibrated intrinsics and optional world placement:

```json
{
  "version": 1,
  "fx_px": 2666.67,
  "fy_px": 2666.67,
  "cx_px": 960,
  "cy_px": 540,
  "sensor_width_mm": 36,
  "sensor_height_mm": 20.25,
  "camera_to_world": [1,0,0,10, 0,1,0,20, 0,0,1,30, 0,0,0,1]
}
```

All four intrinsics must be present together or omitted together (omitted means estimated focal); sensor dimensions and the affine row-major transform are optional. Unknown fields are rejected. Without a pose, P is camera-space metres, +X right/+Y down/+Z forward, pixel centers `(u+.5,v+.5)`; with a pose, P is in world space. Channels are `P.x/P.y/P.z`; nonfinite/nonpositive Z becomes zero P. Camera, units and frame metadata are embedded in every output.

## Embedding

The public surface is `runtime/include/strata.h`; hosts depend only on it, never on ORT or private headers. The header states ownership, lifetime and thread-safety for every type and function. In short:

- `strata_open` creates a context; serialize calls on one context, independent contexts may run concurrently. `strata_unload` frees sessions and scratch, and the next `strata_run` revalidates and reloads. `strata_close` destroys the context; results already returned stay valid.
- Every call that fills an owned output (`strata_result`, `strata_image`, deep, layers, positions) takes a zeroed object or one it filled before, frees the old contents first, and leaves it all zero on failure, so hosts can reuse outputs per frame. Borrowed input buffers must not be an output's own storage; unproject copies its camera and pose first, so those may point into the output.
- `strata_set_head` validates the exported head and sidecar before replacing the configuration; failure leaves the context unchanged; `strata_run_deep` infers both owned DP2 depth and layered deep results. `strata_run` infers one frame into a `strata_result` that owns only the network output; `strata_camera_z_metres` and `strata_inverse_metres` derive metric values. Free with `strata_result_free`. Each result carries its resolved camera, frame index/timestamp, per-stage timings and the count of non-finite input pixels.
- `strata_frame.camera.focal_source` must be `STRATA_FOCAL_PROVIDED` (positive fx/fy) or `STRATA_FOCAL_ESTIMATED` (all intrinsics zero).
- `strata_frame.colorspace` is required; `STRATA_COLOR_UNSET` is rejected. `strata_colorspace_parse` accepts exactly the four CLI names. Both backbone and layered head consume ACEScg; the caller's image is never modified. Deep writers require explicit `frame.colorspace` metadata.
- Images are packed float RGBA rows; the explicit frame colour-space declaration controls inference. `strata_image_read` returns an owned image; a host-filled image keeps `owned = 0` and its buffer.
- Provider `auto` uses CUDA when available, else CPU, and `strata_fallback_reason` says why; `cuda` and `cpu` fail instead of falling back. ONNX Runtime keeps its default allocator and threads; TF32 is disabled. The library never prints.
- In-memory deep buffers support variable counts and packed samples. K-layer extraction takes nearest-Z samples, exposes presence plus alpha-as-coverage, and initializes absent layers to zero RGBA/infinite Z.

## Benchmarking

```sh
gpu_run build/normal/strata_bench --model models/runtime-p0/depthpro2-h.onnx \
  --in beauty.png --in-colorspace lin-rec709 --f-px 2666.67 --provider cuda --frames 10
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.benchmark \
  --bench build/normal/strata_bench --model models/runtime-p0/depthpro2-h.onnx \
  --image beauty.png --in-colorspace lin-rec709 \
  --colour-reference runtime/tests/colour_reference.json --out /path/to/results/performance --engines cuda torch --repeat 10
```

`strata_bench` prints the session time and every frame's stage timings. Cold means a fresh process and session with the filesystem cache left intact; warm frames exclude the first. Both session timers include CUDA initialization but not imports or image reads. Sampled VRAM is the process's own NVIDIA usage, not total device use.

## Colour references

- ACES studio configuration: https://github.com/AcademySoftwareFoundation/OpenColorIO-Config-ACES/releases/tag/v2.1.0-v2.2.0
- Bradford adaptation and primaries: https://github.com/AcademySoftwareFoundation/OpenColorIO/blob/v2.3.0/src/OpenColorIO/transforms/builtins/ColorMatrixHelpers.cpp

## Qualification

`colour` checks the published OCIO matrix values to 1e-6, float32 model RGB exactly, alpha preservation, inverse round-trips and required/unsupported CLI declarations. `colour_mutations` rejects wrong matrices, skipped model conversion, skipped training encoder display conversion and conversion leaking into deep RGB using isolated production copies after a passing baseline. Legacy GPU depth oracles must be regenerated: preprocessing and colour-space identity are now required in every reference NPZ. Committed PNG cases explicitly treat their stored numeric values as linear Rec.709 test signals, not display-encoded inputs.

`layered_parity` in CTest generates a deterministic current-architecture plumbing-only checkpoint and compares every deep sample to a float32 PyTorch oracle on the committed `layered` cases, including both tile axes. Distinct per-channel feature signals and a nontrivial smoke output scale expose tiling errors. It also requires separately created smoke checkpoints to have identical bytes, then checks repeat-export graph identity. CPU and CUDA depth/alpha tolerance is 5e-6 relative and absolute. Plate colour and flatten use 2e-7, total coverage 2e-15. These test plumbing, not model quality. Run CUDA under the shared GPU lock with `training/.venv/bin/python tools/check_layered_head.py --cli build/normal/strata --provider cuda`. `tools/mutate_layered_head.py --build build/cpu` verifies colour, depth, empty-layer, equal-depth, coverage, ordering, both halo axes, core offsets and feature-stride rejection checks in isolated source copies.

GPU jobs are serialized: hold the shared GPU lock, admit a job only with its measured peak plus 2048 MiB free, and run heavy commands under `gpu_run`.

```sh
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.reference \
  --cases runtime/tests/parity_cases.json --colour-reference runtime/tests/colour_reference.json --data-root /path/to/data \
  --out /path/to/reference/fp16 --precision fp16
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.reference \
  --cases runtime/tests/parity_cases.json --colour-reference runtime/tests/colour_reference.json --data-root /path/to/data \
  --out /path/to/reference/fp32 --precision fp32

gpu_run env STRATA_MODEL="$PWD/models/runtime-p0/depthpro2-h.onnx" \
  STRATA_PARITY_DATA=/path/to/data STRATA_PARITY_REFERENCE=/path/to/reference/fp16 \
  STRATA_PARITY_REFERENCE_FP32=/path/to/reference/fp32 \
  tools/check_all.sh
```

Without arguments, `check_all.sh` runs the commit, layer, comment, decision and format checks, their rule cases, Ruff, strict Pyright, Python tests, contract checks and both normal and ASan/UBSan CTest suites; it verifies the cached sanitizer setting so a CMake cache reset cannot silently produce an ordinary build. The parity test reads the committed case list, inputs under `STRATA_PARITY_DATA`, FP16 oracles under `STRATA_PARITY_REFERENCE` and FP32 oracles under `STRATA_PARITY_REFERENCE_FP32`; it skips cleanly when all three are unset and fails when only some are set. Every image gets the same gates on the network inverse depth's relative error, with no per-image exemptions. Against the matching-precision FP16 oracle: 1% at the 99.9th percentile, 0.25% median and 0.25% signed mean, and 5% maximum on pixels more than 2 px from a depth discontinuity (an oracle log inverse-depth step above 0.3 between neighbouring pixels, or a zero boundary); the raw maximum is reported but not gated, because a sub-pixel shift of an occlusion edge gives arbitrary relative error there. Against the FP32 oracle as ground truth: native's median and absolute signed mean may exceed PyTorch FP16's own by at most 0.2 percentage points. Both bounds come from the FP16 noise floor, measured by perturbing 5% of PyTorch FP16's input values by one FP16 ulp; the measurements live with the project results. It requires identical zero masks and checks focal within 0.1%. Before the cases it runs a synthetic 960x544 colour field (`runtime/tests/overflow_field.json`) whose last encoder tap exceeds the FP16 range; the run must succeed with finite, positive inverse depth. Both oracles come from `strata_train.reference` (`--precision fp16` keeps the export's float32 final projection, tap projections and 1/32 decoder branch; `--precision fp32`) with TF32 disabled and are fed bit-identical input tensors.

On Linux only the GPU sanitizer tests set `protect_shadow_gap=0` so CUDA can reserve its address space. `runtime/tests/lsan.supp` suppresses leaks only inside `libcuda.so`, and every sanitized parity run must report exactly that one suppression. The `leak_canary` test runs under the same options and must report a leak allocated by Strata. These sanitizers qualify Strata's host code, not GPU kernels or uninstrumented ORT/NVIDIA internals.

Measurements and qualification transcripts live with the project results, not in this repository.

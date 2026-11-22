# Strata P0 runtime

C11 `libstrata`, the `strata` CLI and the `strata_bench` tool run the single-frame Depth Pro 2 H+ ONNX export. P0 writes metric depth and a **one-sample-per-pixel deep baseline**, not learned hidden surfaces or hair/blur coverage. Linux x86-64 with the CUDA execution provider is the qualified GPU configuration.

## Build

Run commands from the repository root. Install a C11 compiler, CMake >=3.20, Ninja, pkg-config, OpenEXR 3 development files, json-c development files, and clang-format 23.1.4. `stb_image` is vendored with its license. CUDA builds also need the CUDA/cuDNN shared libraries required by the selected ONNX Runtime SDK.

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

Keep all five generated files together: `depthpro2-h.onnx` and `.onnx.weights`, `depthpro2-h.fov.onnx` and its `.weights`, and `depthpro2-h.json`, the exact version-one contract manifest. The exporter finishes with `strata_train.optimize`, which applies ONNX Runtime's basic graph transforms once, rewrites stride==kernel ConvTranspose as a 1x1 Conv plus DepthToSpace and fuses each float32 rotate-half RoPE into one `com.microsoft.RotaryEmbedding`; run `python -m strata_train.optimize model.onnx model.fov.onnx` on a CUDA host to bring an older export up to date. The runtime rejects a missing or incompatible manifest at open and after unload, validates ONNX tensor names/types/shapes, and checks the actual output shape at inference. The manifest describes semantics; it is **not** a cryptographic binding to the weights.

## CLI

```sh
gpu_run build/normal/strata depth \
  --model models/runtime-p0/depthpro2-h.onnx \
  --in plate.exr --f-px 2666.67 --provider cuda \
  --out-depth depth.exr --out-deep deep.exr --out-position P.exr
```

Both depth and deep output paths are required. An output that resolves to an input or to another output (through `.`, `..`, symlinks or hard links) is a usage error. Without `--f-px` or camera intrinsics, focal is estimated by the sibling `.fov.onnx`; a field-of-view output that is not finite and strictly between 0 and 180 degrees fails the run. `--exposure stops` affects scene-linear input's inference conversion, not the preserved deep RGB. `--device N` selects the CUDA device. The CLI prints one JSON object: `provider`, `fallback_reason`, the resolved `focal_px`/`focal_source`, `nonfinite_input_pixels`, `session_ms`, `frame_ms`, `write_ms` and per-stage `stages` (prepare, focal, upload, run, download, postprocess). File reads and writes are outside `frame_ms`.

Inputs are already-oriented 8-bit PNG/JPEG or scanline RGB EXR; tiled, deep and subsampled EXRs are rejected. EXR is scene-linear: expose, replace non-finite samples (+inf to 1, NaN and -inf to 0), clamp to [0,1], encode sRGB, then normalize. Inference runs at native resolution with bottom/right replicate padding to 32 and crops back. The contract allows padded height <=4352 and width <=7680, subject to device memory.

The flat EXR has float32 `Z` and `network_inverse_depth`, ZIP-compressed. For original width `W`, focal `f_px` and network output `q`, `Z = f_px / (W * q)`: optical-axis camera depth in metres, not radial range; zero `q` is `Z=+inf`. Deep EXR has exactly `R G B A Z ZBack`, one sample/pixel, `A=1`, unmodified input RGB, and `ZBack=Z`, as uncompressed float32 samples. This baseline does not recover source alpha, hidden geometry, learned confidence or semantic sky. Every writer renders into a sibling temporary file and renames it over the target.

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
- Every call that fills an owned output (`strata_result`, `strata_image`, deep, layers, positions) takes a zeroed object or one it filled before, frees the old contents first, and leaves it empty on failure, so hosts can reuse outputs per frame. Outputs must not alias inputs.
- `strata_run` infers one frame into a `strata_result` that owns only the network output; `strata_camera_z_metres` and `strata_inverse_metres` derive metric values. Free with `strata_result_free`. Each result carries its resolved camera, frame index/timestamp, per-stage timings and the count of non-finite input pixels.
- `strata_frame.camera.focal_source` must be `STRATA_FOCAL_PROVIDED` (positive fx/fy) or `STRATA_FOCAL_ESTIMATED` (all intrinsics zero).
- Images are packed float RGBA rows with a `linear` flag. `strata_image_read` returns an owned image; a host-filled image keeps `owned = 0` and its buffer.
- Provider `auto` uses CUDA when available, else CPU, and `strata_fallback_reason` says why; `cuda` and `cpu` fail instead of falling back. ONNX Runtime keeps its default allocator and threads; TF32 is disabled. The library never prints.
- In-memory deep buffers support variable counts and packed samples. K-layer extraction takes nearest-Z samples, exposes presence plus alpha-as-coverage, and initializes absent layers to zero RGBA/infinite Z.

## Benchmarking

```sh
gpu_run build/normal/strata_bench --model models/runtime-p0/depthpro2-h.onnx \
  --in beauty.png --f-px 2666.67 --provider cuda --frames 10
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.benchmark \
  --bench build/normal/strata_bench --model models/runtime-p0/depthpro2-h.onnx \
  --image beauty.png --out /path/to/results/performance --engines cuda torch --repeat 10
```

`strata_bench` prints the session time and every frame's stage timings. Cold means a fresh process and session with the filesystem cache left intact; warm frames exclude the first. Both session timers include CUDA initialization but not imports or image reads. Sampled VRAM is the process's own NVIDIA usage, not total device use.

## Qualification

GPU jobs are serialized: hold the shared GPU lock, admit a job only with its measured peak plus 2048 MiB free, and run heavy commands under `gpu_run`.

```sh
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.reference \
  --cases runtime/tests/parity_cases.json --data-root /path/to/data \
  --out /path/to/reference/fp16 --precision fp16
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.reference \
  --cases runtime/tests/parity_cases.json --data-root /path/to/data \
  --out /path/to/reference/fp32 --precision fp32

gpu_run env STRATA_MODEL="$PWD/models/runtime-p0/depthpro2-h.onnx" \
  STRATA_PARITY_DATA=/path/to/data STRATA_PARITY_REFERENCE=/path/to/reference/fp16 \
  STRATA_PARITY_REFERENCE_FP32=/path/to/reference/fp32 \
  tools/check_all.sh
```

Without arguments, `check_all.sh` runs the commit, layer, comment, decision and format checks, their rule cases, Ruff, strict Pyright, Python tests, contract checks and both normal and ASan/UBSan CTest suites; it verifies the cached sanitizer setting so a CMake cache reset cannot silently produce an ordinary build. The parity test reads the committed case list, inputs under `STRATA_PARITY_DATA`, FP16 oracles under `STRATA_PARITY_REFERENCE` and FP32 oracles under `STRATA_PARITY_REFERENCE_FP32`; it skips cleanly when all three are unset and fails when only some are set. Every image gets the same gates on the network inverse depth's relative error, with no per-image exemptions. Against the matching-precision FP16 oracle: 1% at the 99.9th percentile, 0.25% median and 0.25% signed mean, and 5% maximum on pixels more than 2 px from a depth discontinuity (an oracle log inverse-depth step above 0.3 between neighbouring pixels, or a zero boundary); the raw maximum is reported but not gated, because a sub-pixel shift of an occlusion edge gives arbitrary relative error there. Against the FP32 oracle as ground truth: native's median and absolute signed mean may exceed PyTorch FP16's own by at most 0.2 percentage points. Both bounds come from the FP16 noise floor, measured by perturbing 5% of PyTorch FP16's input values by one FP16 ulp; the measurements live with the project results. It requires identical zero masks and checks focal within 0.1%. Both oracles come from `strata_train.reference` (`--precision fp16` keeps the float32 final projection, `--precision fp32`) with TF32 disabled and are fed bit-identical input tensors.

On Linux only the GPU sanitizer tests set `protect_shadow_gap=0` so CUDA can reserve its address space. `runtime/tests/lsan.supp` suppresses leaks only inside `libcuda.so`, and every sanitized parity run must report exactly that one suppression. The `leak_canary` test runs under the same options and must report a leak allocated by Strata. These sanitizers qualify Strata's host code, not GPU kernels or uninstrumented ORT/NVIDIA internals.

Measurements and qualification transcripts live with the project results, not in this repository.

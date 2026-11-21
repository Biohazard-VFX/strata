# Strata P0 runtime

C11 `libstrata` and the `strata` CLI run the single-frame Depth Pro 2 H+ ONNX export. P0 writes metric depth and a **one-sample-per-pixel deep baseline**, not learned hidden surfaces or hair/blur coverage. Linux x86-64 with CUDA is the qualified GPU configuration; other provider paths are not yet qualified.

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

The SDK fetcher pins ONNX Runtime 1.32.0, records download sources/checksums, and refuses a nonempty destination. It tries Microsoft's release archive, then official NuGet headers/PyPI native libraries. To use an existing SDK, pass `-DONNXRUNTIME_ROOT=/absolute/sdk` with `include/onnxruntime_c_api.h` and `lib/libonnxruntime.so`. Binaries, weights and build products are not committed.

For a CPU-only development environment use `--extra cpu`, fetch with `--cpu`, and run `tools/check_all.sh --cpu`. This CPU-only mode runs the unsanitized `cpu` preset with GPU tests excluded; it does not run ASan/UBSan. CPU contract tests use a tiny ONNX fixture; they do not run the full H+ model. `tools/install_hooks.sh` installs the Conventional Commit message hook.

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

Keep all five generated files together:

- `depthpro2-h.onnx` and `depthpro2-h.onnx.weights`;
- `depthpro2-h.fov.onnx` and `depthpro2-h.fov.onnx.weights`;
- `depthpro2-h.json`, the exact version-one contract manifest.

The export uses opset 23 Attention, dynamic tensor-valued RoPE, FP16 learned operations with a float32 final projection, and float32 public tensors. Constant parameter casts are folded before loading. The focal session is separate and lazy. The runtime rejects a missing/incompatible manifest at open and after unload/reload, validates ONNX tensor names/types/shapes, and checks actual output dimensions at inference. The manifest describes semantics; it is **not** a cryptographic binding to the weights. The qualification's separate `export-provenance.json` records artifact/checkpoint hashes.

## CLI

```sh
gpu_run build/normal/strata depth \
  --model models/runtime-p0/depthpro2-h.onnx \
  --in plate.exr --f-px 2666.67 --provider cuda \
  --out-depth depth.exr --out-deep deep.exr --out-position P.exr
```

Both depth and deep output paths are required. Omit `--f-px` to estimate focal from the sibling focal model. `--exposure stops` affects scene-linear input's inference conversion, not the preserved deep RGB. `--device N` selects the GPU; `--repeat N` repeats inference in one session and emits JSON timings. File reads and writes are outside `frame_ms`; the JSON reports `session_ms` and `write_ms` separately.

Inputs are already-oriented 8-bit PNG/JPEG or scanline RGB EXR. Tiled, deep and subsampled EXRs are rejected. EXR is treated as scene-linear: expose, clamp to [0,1], convert to display sRGB, then normalize. Inference uses native resolution with bottom/right replicate padding to 32 and crops back; no implicit 1536-square resize. The contract allows padded height <=4352 and width <=7680, subject to device memory.

Flat EXR channels are float32 `Z`, `network_inverse_depth` and `inverse_metres`, with lossless ZIP compression. For original width `W`, focal `f_px` and network output `q`:

```text
Z = f_px / (W * q)
inverse_metres = W * q / f_px
```

`Z` is optical-axis camera depth in metres, not radial range. Zero `q` stays `Z=+inf` and inverse metres zero. Network inverse depth is not itself inverse metres. Deep EXR has exactly `R G B A Z ZBack`, one sample/pixel, `A=1`, unmodified input RGB, and `ZBack=Z`; it uses uncompressed float32 samples. This baseline does not recover source alpha, hidden geometry, learned confidence or semantic sky.

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

All four intrinsics must be present together or omitted together; sensor dimensions and the affine row-major transform are optional. Unknown fields are rejected. Without a pose, P is camera-space metres, +X right/+Y down/+Z forward; pixel centers are `(u+.5,v+.5)`. With a supplied pose, P is transformed to world space. Channels are `P.x/P.y/P.z`; nonfinite/nonpositive Z becomes zero P (the in-memory API also exposes a validity buffer). No camera pose is estimated. Camera, units and frame metadata are embedded in outputs.

## Embedding

The public surface is `runtime/include/strata.h`; hosts depend only on this API, never ORT or private headers.

- Inputs are tightly packed interleaved float RGBA, row-major, with an explicit `linear` flag; there is no stride argument. `strata_run` currently accepts exactly one frame. Loop unrelated images without losing each result's intrinsics/index/timestamp; do not call a frame loop temporal inference.
- Zero-initialize output structures, free successful allocations with the corresponding `strata_*_free`, and release each result before reusing it. The caller retains the input image. Serialize use of each context and wait for in-flight calls before unloading/closing; there is no cancellation or automatic host memory-manager integration.
- `strata_unload` releases both native sessions; the next run recreates/validates them. `strata_close` destroys the context. Clearing a Torch allocator cache does not unload ORT.
- In-memory deep buffers support variable counts and packed samples. K-layer extraction takes nearest-Z samples, exposes presence plus alpha-as-coverage, and initializes absent layers to zero RGBA/infinite Z. It does not composite opacity bins. Flat, layered and deep unprojection preserve their layouts and optional camera-to-world placement.
- Explicit providers fail when unavailable. `auto` logs its CPU fallback; CUDA sessions retain normal ORT CPU shape operations. `strata_provider` and timing JSON name the configured primary EP, not actual per-node placement. TensorRT also registers CUDA fallback: the benchmark's provider-name check cannot prove any TensorRT engine was built. Any future successful TensorRT configuration run needs independent ORT placement/profiling evidence before being called qualified TensorRT throughput. CUDA has a 13-GiB arena cap, same-as-requested arena growth, disabled TF32 and bounded cuDNN workspace. This arena cap is not a complete process VRAM limit.

## Qualification commands

GPU jobs must be serialized. On the shared workstation acquire the authoritative exclusive `gpu.lock`, post the claim/release on the GPU board thread, and admit each job only with its measured or conservative expected process peak **plus 2048 MiB free**. Never delete another owner's lock or kill desktop processes. Run heavy commands under a user systemd scope with `MemoryMax=32G`, `MemorySwapMax=0`, `timeout`, and `env -u PYTHONPATH`.

```sh
gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.reference \
  --data-root /path/to/deep-depth --manifest /path/to/results/parity.json \
  --out /path/to/results/reference

gpu_run env STRATA_MODEL="$PWD/models/runtime-p0/depthpro2-h.onnx" \
  STRATA_PARITY_MANIFEST=/path/to/results/parity.json \
  STRATA_REFERENCE_DIR=/path/to/results/reference tools/check_all.sh

gpu_run env TORCH_HOME=/path/to/torch-cache \
  training/.venv/bin/python -m strata_train.benchmark \
  --cli build/normal/strata --model models/runtime-p0/depthpro2-h.onnx \
  --image /path/to/beauty.png --out /path/to/results/performance \
  --engines cuda torch torch-stock tensorrt --repeat 10 --timeout 1800
```

Without arguments, `check_all.sh` runs commit/layer/comment/format checks, Ruff, strict Pyright, Python tests, generated-contract checks and both normal and ASan/UBSan CTest suites. It checks the cached sanitizer setting so a compiler-triggered CMake cache reset cannot silently turn the sanitizer build into an ordinary build. The GPU parity test checks all three depth representations, input fingerprints, focal policy and a deep readback. Parity budgets are 5% maximum relative error, 0.5% per-image median, 0.1% pixel-weighted aggregate median and 0.1% focal error; p99 is reported, not gated. Infinity masks must agree. The reference is FP16 with the same float32 final projection, not stock BF16.

On Linux, only the GPU sanitizer test sets `protect_shadow_gap=0` so CUDA can reserve its address space. Leak detection stays enabled; `runtime/tests/lsan.supp` suppresses only allocation stacks containing NVIDIA `cuInit`. A CUDA-only device-count probe, without Strata or ORT, reproduced the same 183 leaked bytes in four allocations. Independent 13-byte leak and use-after-free canaries still failed under these exact GPU options. CPU tests retain ordinary shadow-gap protection and no vendor suppression. These sanitizers qualify Strata's host code, not GPU kernels or uninstrumented ORT/NVIDIA internals.

## Measured results — 2026-11-21

RTX 3090 (24 GiB), NVIDIA driver 615.78.08, Linux x86-64; ONNX Runtime 1.32.0 CUDA EP; PyTorch 2.15.0+cu132. The shared desktop retained approximately 10 GiB of device memory. All GPU jobs held the authoritative exclusive lock and ran in 32-GiB/no-swap systemd scopes. These are rendered-scene implementation checks, not real-plate accuracy or dedicated-GPU throughput claims.

### CUDA parity

The manifest contains 39 beauty images: v001, v003 MB and DOF at 1920×1080, plus 12 sharp, 12 MB and 12 DOF v004 EXRs at 1280×720. All use the same supplied focal of 2666.67 px in both implementations; this is not a ground-truth camera/accuracy evaluation. Errors are `abs(native-reference)/max(abs(reference),1e-6)` on 39,398,400 finite pixels, without scale fitting. Infinity masks agree. Percentages below are pixel-weighted aggregate max / median / p99.

| Image family | Camera-Z error (%) | Network inverse-depth error (%) |
|---|---:|---:|
| v001 | 1.1113 / 0.0132 / 0.1068 | 1.1238 / 0.0132 / 0.1069 |
| v003_mb | 0.7840 / 0.0283 / 0.1671 | 0.7902 / 0.0283 / 0.1674 |
| v003_dof | 2.0508 / 0.0790 / 0.4127 | 2.0937 / 0.0791 / 0.4144 |
| v004_sharp | 3.5439 / 0.0228 / 0.1468 | 3.4226 / 0.0228 / 0.1470 |
| v004_mb | 0.9498 / 0.0338 / 0.1972 | 0.9589 / 0.0338 / 0.1970 |
| v004_dof | 1.1450 / 0.0276 / 0.2967 | 1.1320 / 0.0276 / 0.2958 |
| all | 3.5439 / 0.0285 / 0.3201 | 3.4226 / 0.0285 / 0.3200 |

Inverse-metres aggregate error is **3.4226 / 0.0285 / 0.3200%**. Normal and instrumented builds produced identical aggregate statistics.

The original unfinished 0.1% **per-image** median gate failed on unchanged `dof_05` (0.2545%). Old/new oracle arrays and ONNX operations/weights were identical; disabling Torch TF32 or requesting float32 Attention softmax did not change that discrepancy. The qualified policy now separates a 0.5% per-image median budget from the retained 0.1% aggregate budget and 5% maximum; this is an explicit tolerance-policy change, not a claim that numerical error was repaired. Worst individual-image median is recorded in the per-image JSON.

### Stock BF16 is a separate comparison

| All 39 images, native FP16 vs stock BF16/FP32 head | Max (%) | Median (%) | p99 (%) |
|---|---:|---:|---:|
| Camera-Z | 42.9440 | 0.2030 | 3.0688 |
| Network inverse depth | 31.0142 | 0.2031 | 3.1647 |

The earlier 14.4% maximum described the three 1080p images, not all v004 beauties. The expanded stock comparison reaches 42.9440% camera-Z deviation at `sharp_06`, pixel `(575,399)`: a visually inspected fur/background silhouette, with stock Z=2.3444 m versus native Z=3.3512 m (matching-FP16 reference Z=3.2365 m). Its highest-error 0.1% of pixels all lie within a three-pixel dilation of >5% adjacent reference inverse-depth jumps; that band occupies only 4.59% of this frame. The largest stock inverse-depth deviation is at `mb_06` `(521,0)`, also in this numerical boundary band. Every compared pixel is finite; these are not infinity/invalid-sky disagreements. Changing backbone compute precision and the head autocast policy is not export-only error. No semantic sky classifier was evaluated.

### Performance and sampled VRAM

The same v001 RGB frame is used for every engine, with known focal and ten repeats per fresh process. Warm latency is the median of the nine frames after the first. Both session timers include CUDA initialization but exclude Python imports/image reads; per-frame timing includes normalization, transfer, inference and metric conversion, not EXR writing. Backend handle/algorithm initialization can defer to the first frame differently between engines, so that time is also shown. Session creation is one cold-session observation with OS filesystem caches left intact, not a repeated cold-start distribution; shared-host cache/load effects make the single observations unsuitable for ranking startup speed. Peak VRAM is sampled own-process NVIDIA usage (roughly 0.1-second requested polling plus query overhead), not total device usage or Torch allocated bytes.

| Resolution | Engine / precision | Cold session (ms) | First frame (ms) | Warm frame (ms) | Peak process VRAM (MiB) | Result |
|---|---|---:|---:|---:|---:|---|
| 1920×1080 | ORT CUDA / FP16 | 5493.13 | 983.24 | 588.65 | 6126 | measured |
| 1920×1080 | PyTorch / matching FP16 | 5546.55 | 1306.94 | 531.25 | 9244 | measured |
| 1920×1080 | PyTorch / stock BF16 | 995.92 | 942.43 | 535.76 | 10520 | measured |
| 1920×1080 | ORT TensorRT configuration | — | — | — | — | parser failure / scope OOM |
| 3840×2160 | ORT CUDA / FP16 | — | — | — | — | not launched: headroom |
| 3840×2160 | PyTorch / matching FP16 | — | — | — | — | not launched: headroom |
| 3840×2160 | PyTorch / stock BF16 | — | — | — | — | not launched: headroom |
| 3840×2160 | ORT TensorRT configuration | — | — | — | — | not launched: headroom |

Torch retains upstream FP32 parameter storage while autocasting learned operations; the ONNX export folds FP16 parameter casts. The runtime is not faster than matching PyTorch in this run, but has lower sampled process VRAM.

**4K:** 13,946 MiB was free at admission versus a conservative 14,336-MiB process estimate plus 2,048-MiB reserve (16,384 MiB required). No 4K model job was launched and no 4K timing/peak was invented. This is a shared-workstation qualification limit, not proof that the model cannot run on a dedicated RTX 3090.

**TensorRT:** the installed 10.16.1.11 libraries loaded, but the Attention parser required a strongly typed network that the tested ORT provider did not create. Partition retries were followed by a systemd-scope OOM kill; the journal reports a 23.2-G host-memory peak. This optional path did not build cleanly and remains unqualified. No CUDA fallback result is presented as TensorRT throughput. The attempt used `MemoryMax=32G` and `MemorySwapMax=0`; the recorded 23.2-G peak is below that cap, so the journal establishes an OOM kill but does not establish that the 32-GiB limit itself was reached. The capped failed attempt, parser stderr and journal are retained; third-party ORT/TensorRT sources were not patched.

### Gate transcripts and artifacts

All artifacts are under `/mnt/work/life/projects/deep-depth/results/runtime_p0/`:

- `check-all.log`: full repository checks passed, including Ruff, strict Pyright and three Python tests; normal CTest **11/11**, ASan/UBSan CTest **11/11**, each including all 39 CUDA parity cases.
- `normal-ctest.log`, `sanitize-ctest.log`: complete individual test transcripts; ASan/UBSan symbols and the sanitizer cache were checked.
- `normal-parity/` and `sanitize-parity/`: per-image metrics and aggregate summaries; `parity.json` fingerprints the source cohort through its oracles.
- `export.log`, `export-provenance.json`: final export, manifest validation, graph/external-weight/checkpoint hashes and tensor shapes. The final model is `models/runtime-p0/depthpro2-h.onnx`; graph SHA-256 is `9bfd6458d510bee20a7d913485e332bb66b5486cc92f15dfd1aa02f3e911ae89`.
- `reference/`, `reference-stock/`, `parity-analysis.json`, `analyze_parity.py`, `stock-max-location.png`: independent oracle arrays, spatial error analysis and inspected outlier crop.
- `performance-final/`, `performance-trt4k/`, `qualification.json`: current timings, sampled memory, explicit non-measurements and the consolidated eight engine/resolution outcomes. Original pre-timer-correction runs are retained separately in `performance/`.
- `normal-initial-failure.log`, `check-all-cache-failure.log`, `check-all-shadow-gap-failure.log`, `cuda-init-probe.*`, `sanitizer-canary.log`, `tensorrt-journal.log`: failure/isolation evidence; failures were not replaced by successful-looking output.

Not qualified here: native 4K, TensorRT, non-Linux providers, real-plate/ground-truth accuracy, semantic sky/confidence, learned deep coverage, temporal inference, live host adapters, and exhaustive malformed-media/host-concurrency behavior. The real focal export was generated, but this 39-image GPU cohort uses known focal; focal policy/geometry rejection and lifecycle are covered by independent CPU fixtures.

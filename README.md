# Strata

Single image and video in, deep data out. Part of the Materia toolset from Biohazard VFX.

- `runtime/`: native C inference library and CLI (ONNX Runtime; CUDA/TensorRT, CoreML, DirectML). Embeddable via a C API.
- Nuke nodes and a desktop app (Rust + Rune) build on the runtime.

First milestone: Depth Pro 2 exported to ONNX, run from the C CLI, writing native depth EXR and a direct depth-to-deep EXR baseline.

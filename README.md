# Strata

Deep data from live-action footage.

Strata turns an image or a plate into deep EXRs: several depth samples per pixel, each with its own color and coverage. That lets CG sit behind and in front of hair, fur, foliage, motion blur and defocus without roto. The video model will also solve the camera and recover PBR materials (albedo, normals, roughness, metallic) per layer, for relighting plates in comp.

Status: early development. Not production-ready.

## Layout

- `runtime/`: C library and `strata` CLI on ONNX Runtime (CUDA/TensorRT, CoreML, DirectML). Embeddable through a C API.
- `training/`: model export, post-training and evaluation.
- `contract/`: versioned model spec shared by runtime and training.
- `datagen/`: synthetic dataset generation (USD + MaterialX, rendered in Karma), with deep, cameras and material AOVs per frame.
- `tools/`: repository checks.

## Roadmap

1. Runtime running Depth Pro 2, writing depth plus a direct depth-to-deep baseline.
2. Nuke nodes.
3. Depth Pro 2 post-trained to output deep from a single image.
4. Video model: footage in, deep, camera and per-layer PBR out.
5. Desktop app (Rust + Rune), also on the web.

## Models

The first model is post-trained from Apple's Depth Pro 2. Those weights stay under Apple's research license. Strata's own models will ship under Strata's license.

## License

Elastic License 2.0. Use, modify and fork it freely, including in commercial studio pipelines. You may not offer Strata as a hosted or managed service. Contributions require the CLA (see [CONTRIBUTING.md](CONTRIBUTING.md)).

Part of Materia by Biohazard VFX.

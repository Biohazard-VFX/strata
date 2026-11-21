# Strata

Deep data from live-action footage.

Strata turns an image or a plate into deep EXRs: several depth samples per pixel, each with its own color and coverage. That lets CG sit behind and in front of hair, fur, foliage, motion blur and defocus without roto. Our own model will also solve the camera and recover PBR materials (albedo, normals, roughness, metallic) per layer, for relighting plates in comp.

Status: early development. Not production-ready.

## Layout

- `runtime/`: C library and `strata` CLI on ONNX Runtime (CUDA/TensorRT, CoreML, DirectML). Embeddable through a C API.
- `training/`: model export, post-training and evaluation.
- `contract/`: versioned model spec shared by runtime and training.
- `datagen/`: synthetic dataset generation (USD + MaterialX, rendered in Karma), with deep, cameras and material AOVs per frame.
- `hosts/`: integrations built only on the public C API: Nuke (native deep), OFX, ComfyUI.
- `tools/`: repository checks.

## Roadmap

1. Runtime running Depth Pro 2, writing depth plus a direct depth-to-deep baseline.
2. Host integrations: Nuke nodes first, then OFX and ComfyUI.
3. Depth Pro 2 post-trained to output deep, on single images and video.
4. Our own model, trained from scratch: footage in, deep, camera and per-layer PBR out.
5. Desktop app (Rust + Rune), also on the web.

## Models

The first model is post-trained from Apple's Depth Pro 2. Those weights stay under Apple's research license. Strata's own models will ship under Strata's license.

## References

Strata builds on:

- Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth in Less Than a Second*, 2024. [arXiv:2410.02073](https://arxiv.org/abs/2410.02073)
- Hu et al., *Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth*, 2026. [arXiv:2611.26603](https://arxiv.org/abs/2611.26603)
- Burzio, Fischer et al., *Déjà View: Looping Transformers for Multi-View 3D Reconstruction*, 2026. [arXiv:2605.30215](https://arxiv.org/abs/2605.30215)

BibTeX is in [`CITATION.bib`](CITATION.bib).

## License

Elastic License 2.0. Use, modify and fork it freely, including in commercial studio pipelines. You may not offer Strata as a hosted or managed service. Contributions require the CLA (see [CONTRIBUTING.md](CONTRIBUTING.md)).

Part of Materia by Biohazard VFX.

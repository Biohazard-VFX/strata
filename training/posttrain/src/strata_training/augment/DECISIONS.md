# Decisions

- Augmentations run per sample in the loader from a seeded Generator; nothing is written back to datasets or target caches.
- Grain follows the studio ARRI_Alexa_Grain group node by node; Noise uses a seeded 2048-entry permutation instead of Nuke's table, with matching statistics.
- Grain is generated at input resolution; the group's GrainSample sensor-size resampling (identity at its default) is omitted.
- The group's Blacks/Whites masks and Softness blurs are omitted because their studio defaults are identity.
- Grain, tone, sharpening, quantisation and codecs change only the model input; targets stay untouched.
- Exposure and white balance are scene grades applied to deep sample colours and the input together; input-only gains make colour targets unobservable.
- Camera encodes are Rec.709 (extended-Reinhard highlights, BT.709 OETF) or LogC3 EI800 in AWG3, decoded through their exact inverses.
- Video artefacts come from single-frame intra libx264/libx265 via PyAV with one thread, so bitstreams do not depend on the host.
- Recomposition merges deep samples on one pixel grid; donors must share resolution, may differ in intrinsics (single-image 2.5D merge).
- Lighting mismatch between donor and background is an accepted realism cost, controlled by the recomposition probability.
- Inserted elements bring the donor samples in front of them, so samples Karma dropped as occluded never leave holes.
- Element depth is scaled multiplicatively, preserving its internal log-depth gaps and therefore its K-layer splits.
- Merged inputs split each source's denoised beauty by its merged visible contribution over the source's noisy beauty.
- Synthetic occluders add their exact premultiplied colour; the background's environment is attenuated by every inserted sample.
- Occluder coverage is the exact one-pixel box filter; motion blur and defocus are area-preserving convolutions at one depth.
- Instance and scene grades are 3x3 linear maps on premultiplied colour, so deep compositing algebra is preserved.

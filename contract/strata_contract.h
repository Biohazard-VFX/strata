#ifndef STRATA_CONTRACT_H
#define STRATA_CONTRACT_H

#define STRATA_INPUT_NAME "image"
#define STRATA_OUTPUT_NAME "network_inverse_depth"
#define STRATA_FOV_OUTPUT_NAME "fov_deg"
#define STRATA_FOV_SUFFIX ".fov.onnx"
#define STRATA_INPUT_MULTIPLE 32
#define STRATA_MAX_HEIGHT 4352
#define STRATA_MAX_WIDTH 7680
#define STRATA_FOV_SHORT_MAX 512
#define STRATA_FOV_ASPECT_MAX 1.7778f
#define STRATA_MEAN_R 0.485f
#define STRATA_STD_R 0.229f
#define STRATA_MEAN_G 0.456f
#define STRATA_STD_G 0.224f
#define STRATA_MEAN_B 0.406f
#define STRATA_STD_B 0.225f
#define STRATA_SRGB_THRESHOLD 0.0031308
#define STRATA_SRGB_LINEAR_SCALE 12.92
#define STRATA_SRGB_POWER_SCALE 1.055
#define STRATA_SRGB_POWER_OFFSET 0.055
#define STRATA_SRGB_GAMMA 2.4

#define STRATA_CONTRACT_JSON "{\"contract_version\":2,\"architecture\":\"apple.depthpro2-h.single-frame\",\"opset\":23,\"input\":{\"name\":\"image\",\"dtype\":\"float32\",\"layout\":\"NCHW\",\"shape\":[1,3,\"height\",\"width\"]},\"output\":{\"name\":\"network_inverse_depth\",\"dtype\":\"float32\",\"shape\":[1,1,\"height\",\"width\"],\"semantics\":\"focal-normalized inverse optical-axis depth\"},\"preprocessing\":{\"color\":\"display-sRGB\",\"mean\":[0.485,0.456,0.406],\"std\":[0.229,0.224,0.225],\"padding\":\"bottom-right-replicate\",\"multiple\":32,\"max_height\":4352,\"max_width\":7680,\"linear_exposure\":\"x=max(0,0.72*exp2(shot_stops)*C); x<=0.6 ? x : 0.6+0.4*(x-0.6)/(0.4+x-0.6); sRGB; float32\",\"srgb_threshold\":0.0031308,\"srgb_linear_scale\":12.92,\"srgb_power_scale\":1.055,\"srgb_power_offset\":0.055,\"srgb_gamma\":2.4},\"postprocessing\":{\"crop\":\"original input dimensions\",\"camera_z_metres\":\"focal_px / (original_width * network_inverse_depth)\",\"inverse_metres\":\"original_width * network_inverse_depth / focal_px\",\"zero_network_inverse_depth\":\"camera_z_metres=positive infinity, inverse_metres=0\"},\"fov\":{\"suffix\":\".fov.onnx\",\"input_name\":\"image\",\"input_dtype\":\"float32\",\"input_shape\":[1,3,\"height\",\"width\"],\"output_name\":\"fov_deg\",\"output_dtype\":\"float32\",\"output_shape\":[1,2],\"resize\":\"separable bilinear antialias, half-pixel centers, round ties to even\",\"short_side_max\":512,\"aspect_max\":1.7778,\"focal\":\"0.5 * shorter_original_side / tan(shorter_side_fov_deg * pi / 360)\"},\"deep\":{\"channels\":[\"R\",\"G\",\"B\",\"A\",\"Z\",\"ZBack\"],\"dtype\":\"float32\",\"samples_per_pixel\":\"one per nonempty head layer, at most four, no merging\",\"alpha\":\"conditional coverage with opaque terminal layer\",\"zback\":\"Z\",\"rgb\":\"input RGB multiplied by conditional alpha\",\"alpha_semantics\":\"coverage\",\"rgb_association\":\"premultiplied\"}}"

#endif

#ifndef STRATA_H
#define STRATA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Calls without a context are reentrant on distinct output objects. */

/* Filled outputs: zeroed or a prior output (freed first); empty on failure. */

/* Owns sessions and scratch; serialize calls, contexts are independent. */
typedef struct strata_context strata_context;

/* UNSET is rejected; PROVIDED needs fx/fy, ESTIMATED leaves intrinsics 0. */
enum strata_focal_source {
	STRATA_FOCAL_UNSET,
	STRATA_FOCAL_ESTIMATED,
	STRATA_FOCAL_PROVIDED
};

/* Caller-owned; every failing call writes a NUL-terminated message. */
struct strata_error {
	char message[1024];
};

/* Packed RGBA rows; owned=1 only from strata_image_read, else caller's. */
struct strata_image {
	int width;
	int height;
	int linear;
	float *rgba;
	int owned;
};

/* Pinhole intrinsics in pixels; sensor millimetres are optional, 0 unknown. */
struct strata_camera {
	float fx_px;
	float fy_px;
	float cx_px;
	float cy_px;
	float sensor_width_mm;
	float sensor_height_mm;
	enum strata_focal_source focal_source;
};

/* Borrowed by strata_run for the call only. */
struct strata_frame {
	const struct strata_image *image;
	struct strata_camera camera;
	float exposure_stops;
	int64_t index;
	double timestamp_seconds;
};

/* Copied into every output; the camera is resolved after strata_run. */
struct strata_frame_metadata {
	int64_t index;
	double timestamp_seconds;
	struct strata_camera camera;
	int linear;
};

/* Wall-clock milliseconds per stage of one strata_run call. */
struct strata_timings {
	double prepare_ms;
	double focal_ms;
	double upload_ms;
	double run_ms;
	double download_ms;
	double postprocess_ms;
};

/* Owns network_inverse_depth until strata_result_free; outlives ctx. */
struct strata_result {
	int width;
	int height;
	struct strata_frame_metadata frame;
	float *network_inverse_depth;
	size_t nonfinite_input_pixels;
	struct strata_timings timings;
};

/* Samples packed in pixel order; owns buffers until strata_deep_free. */
struct strata_deep_result {
	int width;
	int height;
	struct strata_frame_metadata frame;
	size_t sample_count;
	uint32_t *counts;
	float *rgba;
	float *z;
	float *zback;
};

/* Layer-major planes, nearest first; owns buffers until layers_free. */
struct strata_layers {
	int width;
	int height;
	struct strata_frame_metadata frame;
	int layer_count;
	uint8_t *present;
	float *rgba;
	float *z;
};

enum strata_coordinate_space { STRATA_SPACE_CAMERA, STRATA_SPACE_WORLD };

enum strata_position_layout {
	STRATA_POSITION_FLAT,
	STRATA_POSITION_LAYERS,
	STRATA_POSITION_DEEP
};

/* Affine row-major camera-to-world matrix on column vectors, metres. */
struct strata_pose {
	float camera_to_world[16];
};

/* Owns xyz and valid until strata_positions_free; invalid points are 0. */
struct strata_positions {
	int width;
	int height;
	int layer_count;
	size_t point_count;
	enum strata_coordinate_space space;
	enum strata_position_layout layout;
	struct strata_frame_metadata frame;
	struct strata_pose pose;
	float *xyz;
	uint8_t *valid;
};

/* Provider "auto", "cuda" or "cpu"; strata_open copies what it keeps. */
struct strata_options {
	const char *model_path;
	const char *provider;
	int device_id;
};

/* On success *out is owned until strata_close; on failure it is NULL. */
int strata_open(const struct strata_options *options, strata_context **out,
		struct strata_error *err);
/* Frees sessions and scratch; the next strata_run revalidates and reloads. */
void strata_unload(strata_context *ctx);
/* Destroys ctx, NULL is a no-op; returned results stay valid. */
void strata_close(strata_context *ctx);
/* Provider in use, "cuda" or "cpu"; borrowed until strata_close. */
const char *strata_provider(const strata_context *ctx);
/* Why auto fell back to CPU, or ""; borrowed until strata_close. */
const char *strata_fallback_reason(const strata_context *ctx);
/* Fills result; see the filled-output rule above. */
int strata_run(strata_context *ctx, const struct strata_frame *frame,
	       struct strata_result *result, struct strata_error *err);
/* Metric values derived from one pixel of the network output. */
float strata_camera_z_metres(const struct strata_result *result, size_t pixel);
float strata_inverse_metres(const struct strata_result *result, size_t pixel);
void strata_result_free(struct strata_result *result);
/* Fills an owned image; release it with strata_image_free. */
int strata_image_read(const char *path, struct strata_image *image,
		      struct strata_error *err);
/* Frees rgba only when owned; always zeroes the struct. */
void strata_image_free(struct strata_image *image);
/* Reads version-1 camera JSON; *has_pose reports an optional pose. */
int strata_camera_read(const char *path, struct strata_camera *camera,
		       struct strata_pose *pose, int *has_pose,
		       struct strata_error *err);
/* Fill positions; a NULL camera uses the input's resolved camera. */
int strata_unproject_depth(const struct strata_result *depth,
			   const struct strata_camera *camera,
			   const struct strata_pose *pose,
			   struct strata_positions *positions,
			   struct strata_error *err);
int strata_unproject_layers(const struct strata_layers *layers,
			    const struct strata_camera *camera,
			    const struct strata_pose *pose,
			    struct strata_positions *positions,
			    struct strata_error *err);
int strata_unproject_deep(const struct strata_deep_result *deep,
			  const struct strata_camera *camera,
			  const struct strata_pose *pose,
			  struct strata_positions *positions,
			  struct strata_error *err);
void strata_positions_free(struct strata_positions *positions);
/* Fills deep with copies of the image RGB and derived camera-Z. */
int strata_depth_to_deep(const struct strata_image *image,
			 const struct strata_result *depth,
			 struct strata_deep_result *deep,
			 struct strata_error *err);
void strata_deep_free(struct strata_deep_result *deep);
/* Fills layers with the nearest count samples of each pixel. */
int strata_deep_to_layers(const struct strata_deep_result *deep, int count,
			  struct strata_layers *layers,
			  struct strata_error *err);
void strata_layers_free(struct strata_layers *layers);
/* Sets *same if a and b resolve to one file, existing or not; reentrant. */
int strata_same_file(const char *a, const char *b, int *same,
		     struct strata_error *err);
/* Writers replace path atomically and leave it untouched on failure. */
int strata_write_depth(const char *path, const struct strata_result *result,
		       struct strata_error *err);
int strata_write_deep(const char *path, const struct strata_deep_result *deep,
		      struct strata_error *err);
int strata_write_position(const char *path,
			  const struct strata_positions *positions,
			  struct strata_error *err);

#ifdef __cplusplus
}
#endif
#endif

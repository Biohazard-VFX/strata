#ifndef STRATA_H
#define STRATA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct strata_context strata_context;

enum strata_focal_source { STRATA_FOCAL_ESTIMATED, STRATA_FOCAL_PROVIDED };

struct strata_error {
	char message[1024];
};

struct strata_image {
	int width;
	int height;
	int linear;
	float *rgba;
};

struct strata_camera {
	float fx_px;
	float fy_px;
	float cx_px;
	float cy_px;
	float sensor_width_mm;
	float sensor_height_mm;
	enum strata_focal_source focal_source;
};

struct strata_frame {
	const struct strata_image *image;
	struct strata_camera camera;
	float exposure_stops;
	int64_t index;
	double timestamp_seconds;
};

struct strata_frame_metadata {
	int64_t index;
	double timestamp_seconds;
	struct strata_camera camera;
	int linear;
};

struct strata_result {
	int width;
	int height;
	struct strata_frame_metadata frame;
	float *network_inverse_depth;
	float *inverse_metres;
	float *camera_z_metres;
};

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

struct strata_pose {
	float camera_to_world[16];
};

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

struct strata_options {
	const char *model_path;
	const char *provider;
	int device_id;
};

int strata_open(const struct strata_options *options, strata_context **out,
		struct strata_error *err);
void strata_unload(strata_context *ctx);
void strata_close(strata_context *ctx);
const char *strata_provider(const strata_context *ctx);
int strata_run(strata_context *ctx, const struct strata_frame *frames,
	       size_t frame_count, struct strata_result *results,
	       struct strata_error *err);
int strata_image_read(const char *path, struct strata_image *image,
		      struct strata_error *err);
void strata_image_free(struct strata_image *image);
void strata_result_free(struct strata_result *result);
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
int strata_write_position(const char *path,
			  const struct strata_positions *positions,
			  struct strata_error *err);
int strata_camera_read(const char *path, struct strata_camera *camera,
		       struct strata_pose *pose, int *has_pose,
		       struct strata_error *err);
int strata_write_depth(const char *path, const struct strata_result *result,
		       struct strata_error *err);
int strata_depth_to_deep(const struct strata_image *image,
			 const struct strata_result *depth,
			 struct strata_deep_result *deep,
			 struct strata_error *err);
void strata_deep_free(struct strata_deep_result *deep);
int strata_deep_to_layers(const struct strata_deep_result *deep, int count,
			  struct strata_layers *layers,
			  struct strata_error *err);
void strata_layers_free(struct strata_layers *layers);
int strata_write_deep(const char *path, const struct strata_deep_result *deep,
		      struct strata_error *err);

#ifdef __cplusplus
}
#endif
#endif

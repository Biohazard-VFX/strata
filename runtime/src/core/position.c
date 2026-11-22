#include "internal.h"
#include <float.h>

void strata_positions_free(struct strata_positions *positions)
{
	if (!positions)
		return;
	free(positions->xyz);
	free(positions->valid);
	memset(positions, 0, sizeof(*positions));
}

int strata_camera_validate(const struct strata_camera *camera, int resolved,
			   struct strata_error *err)
{
	int unresolved;

	if (!camera)
		return strata_fail(err, "missing camera");
	if (camera->focal_source != STRATA_FOCAL_ESTIMATED &&
	    camera->focal_source != STRATA_FOCAL_PROVIDED)
		return strata_fail(
			err, "camera focal source must be set explicitly");
	if (!isfinite(camera->fx_px) || !isfinite(camera->fy_px) ||
	    !isfinite(camera->cx_px) || !isfinite(camera->cy_px) ||
	    !isfinite(camera->sensor_width_mm) || camera->sensor_width_mm < 0 ||
	    !isfinite(camera->sensor_height_mm) || camera->sensor_height_mm < 0)
		return strata_fail(err, "invalid camera intrinsics");
	unresolved = !resolved &&
		     camera->focal_source == STRATA_FOCAL_ESTIMATED;
	if (unresolved && (camera->fx_px != 0 || camera->fy_px != 0 ||
			   camera->cx_px != 0 || camera->cy_px != 0))
		return strata_fail(err, "estimated focal takes no intrinsics");
	if (!unresolved && (camera->fx_px <= 0 || camera->fy_px <= 0))
		return strata_fail(err,
				   "camera focal lengths must be positive");
	return 0;
}

int strata_pose_validate(const struct strata_pose *pose,
			 struct strata_error *err)
{
	const float *matrix;

	if (!pose)
		return 0;
	matrix = pose->camera_to_world;
	for (size_t i = 0; i < 16; ++i) {
		if (!isfinite(matrix[i]))
			return strata_fail(err,
					   "non-finite camera-to-world pose");
	}
	if (matrix[12] != 0 || matrix[13] != 0 || matrix[14] != 0 ||
	    matrix[15] != 1)
		return strata_fail(
			err, "camera-to-world pose must be affine, row-major");
	return 0;
}

static int allocate_positions(int w, int h, size_t count,
			      const struct strata_frame_metadata *frame,
			      const struct strata_camera *camera,
			      const struct strata_pose *pose,
			      struct strata_positions *positions,
			      struct strata_error *err)
{
	struct strata_positions out = { 0 };

	if (strata_dimensions(w, h, err) || strata_pose_validate(pose, err) ||
	    strata_camera_validate(camera ? camera : &frame->camera, 1, err))
		return -1;
	if (count > SIZE_MAX / (3 * sizeof(float)))
		return strata_fail(err, "position allocation overflow");
	out.width = w;
	out.height = h;
	out.layer_count = 1;
	out.point_count = count;
	out.space = pose ? STRATA_SPACE_WORLD : STRATA_SPACE_CAMERA;
	out.frame = *frame;
	if (camera)
		out.frame.camera = *camera;
	if (pose)
		out.pose = *pose;
	else {
		for (size_t i = 0; i < 4; ++i)
			out.pose.camera_to_world[i * 5] = 1;
	}
	out.xyz = calloc(count ? count : 1, 3 * sizeof(float));
	out.valid = calloc(count ? count : 1, sizeof(uint8_t));
	if (!out.xyz || !out.valid) {
		strata_positions_free(&out);
		return strata_fail(err, "out of memory");
	}
	*positions = out;
	return 0;
}

static void unproject(struct strata_positions *positions, size_t point,
		      size_t pixel, float z)
{
	const struct strata_camera *camera = &positions->frame.camera;
	const float *matrix = positions->pose.camera_to_world;
	size_t x = pixel % (size_t)positions->width;
	size_t y = pixel / (size_t)positions->width;
	double camera_point[3], output[3];

	if (!isfinite(z) || z <= 0)
		return;
	camera_point[0] = ((double)x + .5 - camera->cx_px) * z / camera->fx_px;
	camera_point[1] = ((double)y + .5 - camera->cy_px) * z / camera->fy_px;
	camera_point[2] = z;
	for (size_t c = 0; c < 3; ++c) {
		output[c] = matrix[c * 4] * camera_point[0] +
			    matrix[c * 4 + 1] * camera_point[1] +
			    matrix[c * 4 + 2] * camera_point[2] +
			    matrix[c * 4 + 3];
		if (!isfinite(output[c]) || fabs(output[c]) > FLT_MAX)
			return;
	}
	for (size_t c = 0; c < 3; ++c)
		positions->xyz[point * 3 + c] = (float)output[c];
	positions->valid[point] = 1;
}

int strata_unproject_depth(const struct strata_result *depth,
			   const struct strata_camera *camera,
			   const struct strata_pose *pose,
			   struct strata_positions *positions,
			   struct strata_error *err)
{
	struct strata_result view;
	size_t count;

	if (!positions)
		return strata_fail(err, "missing position output");
	strata_positions_free(positions);
	if (!depth || !depth->network_inverse_depth)
		return strata_fail(err, "missing network inverse depth");
	if (strata_dimensions(depth->width, depth->height, err))
		return -1;
	count = (size_t)depth->width * (size_t)depth->height;
	if (allocate_positions(depth->width, depth->height, count,
			       &depth->frame, camera, pose, positions, err))
		return -1;
	positions->layout = STRATA_POSITION_FLAT;
	view = *depth;
	view.frame.camera = positions->frame.camera;
	for (size_t i = 0; i < count; ++i)
		unproject(positions, i, i, strata_camera_z_metres(&view, i));
	return 0;
}

int strata_unproject_layers(const struct strata_layers *layers,
			    const struct strata_camera *camera,
			    const struct strata_pose *pose,
			    struct strata_positions *positions,
			    struct strata_error *err)
{
	size_t pixels, count;

	if (!positions)
		return strata_fail(err, "missing position output");
	strata_positions_free(positions);
	if (!layers || !layers->z || !layers->present ||
	    layers->layer_count < 1)
		return strata_fail(err, "missing layer depth/presence input");
	if (strata_dimensions(layers->width, layers->height, err))
		return -1;
	pixels = (size_t)layers->width * (size_t)layers->height;
	if (pixels > SIZE_MAX / (size_t)layers->layer_count)
		return strata_fail(err, "layer position allocation overflow");
	count = pixels * (size_t)layers->layer_count;
	if (allocate_positions(layers->width, layers->height, count,
			       &layers->frame, camera, pose, positions, err))
		return -1;
	positions->layout = STRATA_POSITION_LAYERS;
	positions->layer_count = layers->layer_count;
	for (size_t i = 0; i < count; ++i) {
		if (layers->present[i])
			unproject(positions, i, i % pixels, layers->z[i]);
	}
	return 0;
}

int strata_unproject_deep(const struct strata_deep_result *deep,
			  const struct strata_camera *camera,
			  const struct strata_pose *pose,
			  struct strata_positions *positions,
			  struct strata_error *err)
{
	size_t pixels, offset = 0;

	if (!positions)
		return strata_fail(err, "missing position output");
	strata_positions_free(positions);
	if (strata_deep_validate(deep, err))
		return -1;
	if (allocate_positions(deep->width, deep->height, deep->sample_count,
			       &deep->frame, camera, pose, positions, err))
		return -1;
	positions->layout = STRATA_POSITION_DEEP;
	positions->layer_count = 0;
	pixels = (size_t)deep->width * (size_t)deep->height;
	for (size_t pixel = 0; pixel < pixels; ++pixel) {
		for (size_t sample = 0; sample < deep->counts[pixel]; ++sample)
			unproject(positions, offset + sample, pixel,
				  deep->z[offset + sample]);
		offset += deep->counts[pixel];
	}
	return 0;
}

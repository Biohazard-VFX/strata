#include "internal.h"
#include <openexr.h>
#include <inttypes.h>
#include <OpenEXRConfig.h>

static void quiet_exr(exr_const_context_t context, exr_result_t code,
		      const char *message)
{
	(void)context;
	(void)code;
	(void)message;
}

static const exr_context_initializer_t exr_options = {
	.size = sizeof(exr_context_initializer_t),
	.error_handler_fn = quiet_exr,
};

static int exr_failed(exr_result_t rc, const char *what,
		      struct strata_error *err)
{
	if (rc == EXR_ERR_SUCCESS)
		return 0;
	return strata_fail(err, "OpenEXR: %s (%s)",
			   exr_get_error_code_as_string(rc), what);
}

static int decode_channels(exr_decode_pipeline_t *pipe,
			   struct strata_image *image, int row,
			   struct strata_error *err)
{
	unsigned int found = 0;

	for (int c = 0; c < pipe->channel_count; ++c) {
		exr_coding_channel_info_t *channel = &pipe->channels[c];
		const char *name = channel->channel_name;
		int component = !strcmp(name, "R") ? 0 :
				!strcmp(name, "G") ? 1 :
				!strcmp(name, "B") ? 2 :
						     -1;

		channel->decode_to_ptr = NULL;
		if (component < 0)
			continue;
		if (channel->x_samples != 1 || channel->y_samples != 1)
			return strata_fail(err, "subsampled RGB unsupported");
		channel->user_data_type = EXR_PIXEL_FLOAT;
		channel->user_bytes_per_element = 4;
		channel->user_pixel_stride = 16;
		channel->user_line_stride = image->width * 16;
		channel->decode_to_ptr =
			(uint8_t *)(image->rgba +
				    (size_t)row * (size_t)image->width * 4 +
				    (size_t)component);
		found |= 1u << (unsigned int)component;
	}
	return found == 7 ?
		       0 :
		       strata_fail(err,
				   "EXR requires first-part R G B channels");
}

static int allocate_exr(exr_context_t ctx, struct strata_image *image,
			exr_attr_box2i_t *window, struct strata_error *err)
{
	exr_storage_t storage;
	int64_t width, height;
	size_t pixels;

	if (exr_failed(exr_get_storage(ctx, 0, &storage), "storage", err))
		return -1;
	if (storage != EXR_STORAGE_SCANLINE)
		return strata_fail(err,
				   "input EXR must be a flat scanline image");
	if (exr_failed(exr_get_data_window(ctx, 0, window), "data window", err))
		return -1;
	width = (int64_t)window->max.x - window->min.x + 1;
	height = (int64_t)window->max.y - window->min.y + 1;
	if (width > INT_MAX || width < 1 || height > INT_MAX || height < 1)
		return strata_fail(err, "invalid EXR data window");
	if (strata_dimensions((int)width, (int)height, err))
		return -1;
	pixels = (size_t)width * (size_t)height;
	image->rgba = calloc(pixels * 4, sizeof(float));
	if (!image->rgba)
		return strata_fail(err, "out of memory");
	image->width = (int)width;
	image->height = (int)height;
	image->linear = 1;
	image->owned = 1;
	for (size_t i = 0; i < pixels; ++i)
		image->rgba[i * 4 + 3] = 1;
	return 0;
}

static int decode_chunk(exr_context_t ctx, exr_decode_pipeline_t *pipe,
			struct strata_image *image, int y, int *row,
			struct strata_error *err)
{
	exr_chunk_info_t chunk;
	exr_result_t rc;

	if (exr_failed(exr_read_scanline_chunk_info(ctx, 0, y, &chunk), "chunk",
		       err))
		return -1;
	rc = *row ? exr_decoding_update(ctx, 0, &chunk, pipe) :
		    exr_decoding_initialize(ctx, 0, &chunk, pipe);
	if (exr_failed(rc, "decoder", err) ||
	    decode_channels(pipe, image, *row, err) ||
	    exr_failed(exr_decoding_choose_default_routines(ctx, 0, pipe),
		       "decoder", err) ||
	    exr_failed(exr_decoding_run(ctx, 0, pipe), "decode", err))
		return -1;
	*row += chunk.height;
	return 0;
}

int strata_read_exr(const char *path, struct strata_image *image,
		    struct strata_error *err)
{
	exr_context_t ctx = NULL;
	exr_decode_pipeline_t pipe = { 0 };
	exr_attr_box2i_t window;
	int ret = -1, row = 0;

#if OPENEXR_VERSION_MAJOR > 3 || OPENEXR_VERSION_MINOR >= 3
	pipe.pipe_size = sizeof(pipe);
#endif
	if (exr_failed(exr_start_read(&ctx, path, &exr_options), path, err) ||
	    allocate_exr(ctx, image, &window, err))
		goto out;
	while (row < image->height) {
		if (decode_chunk(ctx, &pipe, image, window.min.y + row, &row,
				 err))
			goto out;
	}
	ret = 0;
out:
	if (ctx) {
		exr_decoding_destroy(ctx, &pipe);
		if (exr_finish(&ctx) != EXR_ERR_SUCCESS && !ret)
			ret = strata_fail(err, "OpenEXR close failed");
	}
	if (ret)
		strata_image_free(image);
	return ret;
}

static void little32(void *dst, uint32_t value)
{
	uint8_t *bytes = dst;

	for (unsigned int i = 0; i < 4; ++i)
		bytes[i] = (uint8_t)(value >> (8 * i));
}

static int write_metadata(exr_context_t ctx, int deep,
			  const struct strata_frame_metadata *frame,
			  struct strata_error *err)
{
	const struct strata_camera *camera = &frame->camera;
	char index[32];
	const char *strings[][2] = {
		{ "strata:frame_index", index },
		{ "strata:depth_units", "metres" },
		{ "strata:depth_kind", "camera-Z" },
		{ "strata:focal_source",
		  camera->focal_source == STRATA_FOCAL_PROVIDED ? "provided" :
								  "estimated" },
		{ "strata:rgb_association", "premultiplied" },
		{ "strata:alpha_semantics", "coverage" },
		{ "strata:color_space",
		  frame->linear ? "scene-linear" : "display-sRGB" },
	};
	const char *names[] = {
		"strata:fx_px",		  "strata:fy_px",
		"strata:cx_px",		  "strata:cy_px",
		"strata:sensor_width_mm", "strata:sensor_height_mm"
	};
	const float values[] = {
		camera->fx_px, camera->fy_px,		camera->cx_px,
		camera->cy_px, camera->sensor_width_mm, camera->sensor_height_mm
	};

	snprintf(index, sizeof(index), "%" PRId64, frame->index);
	for (size_t i = 0; i < (deep ? 7u : 4u); ++i) {
		if (exr_failed(exr_attr_set_string(ctx, 0, strings[i][0],
						   strings[i][1]),
			       strings[i][0], err))
			return -1;
	}
	for (size_t i = 0; i < 6; ++i) {
		if (exr_failed(exr_attr_set_float(ctx, 0, names[i], values[i]),
			       names[i], err))
			return -1;
	}
	return exr_failed(exr_attr_set_double(ctx, 0,
					      "strata:timestamp_seconds",
					      frame->timestamp_seconds),
			  "timestamp", err);
}

enum exr_kind { STRATA_EXR_DEPTH, STRATA_EXR_DEEP, STRATA_EXR_POSITION };

static int write_header(exr_context_t ctx, int w, int h, enum exr_kind kind,
			const struct strata_frame_metadata *frame,
			struct strata_error *err)
{
	const char *deep_names[] = { "A", "B", "G", "R", "Z", "ZBack" };
	const char *depth_names[] = { "Z", "network_inverse_depth" };
	const char *position_names[] = { "P.x", "P.y", "P.z" };
	int deep = kind == STRATA_EXR_DEEP;
	const char *const *names = deep ? deep_names :
				   kind == STRATA_EXR_POSITION ?
					  position_names :
					  depth_names;
	size_t count = deep ? 6u : kind == STRATA_EXR_POSITION ? 3u : 2u;
	int part;

	if (exr_failed(exr_add_part(ctx, NULL,
				    deep ? EXR_STORAGE_DEEP_SCANLINE :
					   EXR_STORAGE_SCANLINE,
				    &part),
		       "part", err) ||
	    exr_failed(exr_initialize_required_attr_simple(
			       ctx, part, w, h,
			       deep ? EXR_COMPRESSION_NONE :
				      EXR_COMPRESSION_ZIP),
		       "header", err) ||
	    (deep && exr_failed(exr_set_version(ctx, part, 1), "version", err)))
		return -1;
	for (size_t c = 0; c < count; ++c) {
		if (exr_failed(exr_add_channel(ctx, part, names[c],
					       EXR_PIXEL_FLOAT,
					       EXR_PERCEPTUALLY_LINEAR, 1, 1),
			       names[c], err))
			return -1;
	}
	return write_metadata(ctx, deep, frame, err);
}

struct exr_plane {
	const char *name;
	const float *data;
	int stride;
};

static const struct exr_plane *find_plane(const struct exr_plane *planes,
					  size_t count, const char *name)
{
	for (size_t i = 0; i < count; ++i) {
		if (!strcmp(name, planes[i].name))
			return &planes[i];
	}
	return NULL;
}

static int bind_planes(exr_encode_pipeline_t *pipe, int width, int row,
		       const struct exr_plane *planes, size_t count,
		       struct strata_error *err)
{
	for (int c = 0; c < pipe->channel_count; ++c) {
		exr_coding_channel_info_t *channel = &pipe->channels[c];
		const struct exr_plane *plane =
			find_plane(planes, count, channel->channel_name);

		if (!plane)
			return strata_fail(err, "unmapped EXR output channel");
		channel->user_data_type = EXR_PIXEL_FLOAT;
		channel->user_bytes_per_element = 4;
		channel->user_pixel_stride = plane->stride * 4;
		channel->user_line_stride = width * plane->stride * 4;
		channel->encode_from_ptr =
			(const uint8_t *)(plane->data +
					  (size_t)row * (size_t)width *
						  (size_t)plane->stride);
	}
	return 0;
}

static int write_flat(exr_context_t ctx, int width, int height,
		      const struct exr_plane *planes, size_t count,
		      struct strata_error *err)
{
	exr_encode_pipeline_t pipe = { 0 };
	int row = 0, ret = -1;

	while (row < height) {
		exr_chunk_info_t chunk;
		exr_result_t rc;

		if (exr_failed(exr_write_scanline_chunk_info(ctx, 0, row,
							     &chunk),
			       "chunk", err))
			goto out;
		rc = row ? exr_encoding_update(ctx, 0, &chunk, &pipe) :
			   exr_encoding_initialize(ctx, 0, &chunk, &pipe);
		if (exr_failed(rc, "encoder", err) ||
		    bind_planes(&pipe, width, row, planes, count, err) ||
		    exr_failed(exr_encoding_choose_default_routines(ctx, 0,
								    &pipe),
			       "encoder", err) ||
		    exr_failed(exr_encoding_run(ctx, 0, &pipe), "encode", err))
			goto out;
		row += chunk.height;
	}
	ret = 0;
out:
	exr_encoding_destroy(ctx, &pipe);
	return ret;
}

static size_t row_samples(const struct strata_deep_result *deep, size_t row)
{
	size_t total = 0;
	size_t width = (size_t)deep->width;

	for (size_t x = 0; x < width; ++x)
		total += deep->counts[row * width + x];
	return total;
}

static void deep_row(float *planes, const struct strata_deep_result *deep,
		     size_t offset, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		size_t sample = offset + i;
		float values[] = { deep->rgba[sample * 4 + 3],
				   deep->rgba[sample * 4 + 2],
				   deep->rgba[sample * 4 + 1],
				   deep->rgba[sample * 4],
				   deep->z[sample],
				   deep->zback[sample] };

		for (size_t c = 0; c < 6; ++c) {
			uint32_t bits;

			memcpy(&bits, &values[c], sizeof(bits));
			little32(planes + c * count + i, bits);
		}
	}
}

static int write_deep(exr_context_t ctx, const struct strata_deep_result *deep,
		      struct strata_error *err)
{
	size_t maximum = 0, offset = 0, width = (size_t)deep->width;
	float *planes = NULL;
	int32_t *counts = NULL;
	int ret = -1;

	for (size_t row = 0; row < (size_t)deep->height; ++row) {
		size_t samples = row_samples(deep, row);

		if (samples > INT32_MAX ||
		    samples > SIZE_MAX / (6 * sizeof(float)))
			return strata_fail(
				err,
				"deep scanline sample count exceeds EXR limits");
		if (samples > maximum)
			maximum = samples;
	}
	planes = malloc((maximum ? maximum : 1) * 6 * sizeof(float));
	counts = malloc(width * sizeof(int32_t));
	if (!planes || !counts) {
		strata_fail(err, "out of memory");
		goto out;
	}
	for (int row = 0; row < deep->height; ++row) {
		uint32_t samples = 0;

		for (size_t x = 0; x < width; ++x) {
			samples += deep->counts[(size_t)row * width + x];
			little32(counts + x, samples);
		}
		deep_row(planes, deep, offset, samples);
		if (exr_failed(exr_write_deep_scanline_chunk(
				       ctx, 0, row, planes,
				       (uint64_t)24 * samples,
				       (uint64_t)24 * samples, counts,
				       (uint64_t)4 * width),
			       "deep chunk", err))
			goto out;
		offset += samples;
	}
	ret = 0;
out:
	free(planes);
	free(counts);
	return ret;
}

static int open_output(const char *path, exr_context_t *ctx, char **temporary,
		       int w, int h, enum exr_kind kind,
		       const struct strata_frame_metadata *frame,
		       struct strata_error *err)
{
	if (strata_atomic_target(path, temporary, err) ||
	    exr_failed(exr_start_write(ctx, *temporary, EXR_WRITE_FILE_DIRECTLY,
				       &exr_options),
		       path, err))
		return -1;
	return write_header(*ctx, w, h, kind, frame, err);
}

static int close_output(const char *path, exr_context_t *ctx, char *temporary,
			int ret, struct strata_error *err)
{
	if (*ctx && exr_finish(ctx) != EXR_ERR_SUCCESS && !ret)
		ret = strata_fail(err, "OpenEXR close failed");
	if (!temporary)
		return ret;
	return strata_atomic_finish(path, temporary, ret, err);
}

int strata_write_depth(const char *path, const struct strata_result *result,
		       struct strata_error *err)
{
	exr_context_t ctx = NULL;
	char *temporary = NULL;
	struct exr_plane planes[2];
	float *z;
	size_t pixels;
	int ret = -1;

	if (!path || !result || !result->network_inverse_depth)
		return strata_fail(err, "invalid EXR output arguments");
	if (strata_dimensions(result->width, result->height, err) ||
	    strata_camera_validate(&result->frame.camera, 1, err))
		return -1;
	pixels = (size_t)result->width * (size_t)result->height;
	z = malloc(pixels * sizeof(float));
	if (!z)
		return strata_fail(err, "out of memory");
	for (size_t i = 0; i < pixels; ++i)
		z[i] = strata_camera_z_metres(result, i);
	planes[0] = (struct exr_plane){ "Z", z, 1 };
	planes[1] = (struct exr_plane){ "network_inverse_depth",
					result->network_inverse_depth, 1 };
	if (open_output(path, &ctx, &temporary, result->width, result->height,
			STRATA_EXR_DEPTH, &result->frame, err) ||
	    exr_failed(exr_write_header(ctx), path, err))
		goto out;
	ret = write_flat(ctx, result->width, result->height, planes, 2, err);
out:
	free(z);
	return close_output(path, &ctx, temporary, ret, err);
}

int strata_write_deep(const char *path, const struct strata_deep_result *deep,
		      struct strata_error *err)
{
	exr_context_t ctx = NULL;
	char *temporary = NULL;
	int ret = -1;

	if (!path)
		return strata_fail(err, "missing EXR output path");
	if (strata_deep_validate(deep, err))
		return -1;
	if (open_output(path, &ctx, &temporary, deep->width, deep->height,
			STRATA_EXR_DEEP, &deep->frame, err) ||
	    exr_failed(exr_write_header(ctx), path, err))
		goto out;
	ret = write_deep(ctx, deep, err);
out:
	return close_output(path, &ctx, temporary, ret, err);
}

static int position_attributes(exr_context_t ctx,
			       const struct strata_positions *positions,
			       struct strata_error *err)
{
	exr_attr_m44f_t matrix;

	memcpy(matrix.m, positions->pose.camera_to_world, sizeof(matrix.m));
	if (exr_failed(exr_attr_set_string(ctx, 0, "strata:P_space",
					   positions->space ==
							   STRATA_SPACE_WORLD ?
						   "world" :
						   "camera"),
		       "P space", err) ||
	    exr_failed(exr_attr_set_string(ctx, 0, "strata:P_invalid",
					   "zero XYZ"),
		       "P invalid", err) ||
	    exr_failed(exr_attr_set_string(ctx, 0, "strata:pose_convention",
					   "row-major, column vectors, metres"),
		       "pose convention", err) ||
	    exr_failed(exr_attr_set_m44f(ctx, 0, "strata:camera_to_world",
					 &matrix),
		       "pose", err))
		return -1;
	return 0;
}

int strata_write_position(const char *path,
			  const struct strata_positions *positions,
			  struct strata_error *err)
{
	exr_context_t ctx = NULL;
	char *temporary = NULL;
	struct exr_plane planes[3];
	int ret = -1;

	if (!path || !positions || !positions->xyz ||
	    positions->layer_count != 1 ||
	    positions->layout == STRATA_POSITION_DEEP ||
	    (positions->space != STRATA_SPACE_CAMERA &&
	     positions->space != STRATA_SPACE_WORLD))
		return strata_fail(
			err, "position EXR requires one flat/front layer");
	if (strata_dimensions(positions->width, positions->height, err))
		return -1;
	if (positions->point_count !=
	    (size_t)positions->width * (size_t)positions->height)
		return strata_fail(
			err, "position count does not match image dimensions");
	planes[0] = (struct exr_plane){ "P.x", positions->xyz, 3 };
	planes[1] = (struct exr_plane){ "P.y", positions->xyz + 1, 3 };
	planes[2] = (struct exr_plane){ "P.z", positions->xyz + 2, 3 };
	if (open_output(path, &ctx, &temporary, positions->width,
			positions->height, STRATA_EXR_POSITION,
			&positions->frame, err) ||
	    position_attributes(ctx, positions, err) ||
	    exr_failed(exr_write_header(ctx), path, err))
		goto out;
	ret = write_flat(ctx, positions->width, positions->height, planes, 3,
			 err);
out:
	return close_output(path, &ctx, temporary, ret, err);
}

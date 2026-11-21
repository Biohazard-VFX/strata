#include "internal.h"
#include <openexr.h>
#include <inttypes.h>
#include <OpenEXRConfig.h>

#define EXR(call)                                                             \
	do {                                                                  \
		exr_result_t rc = (call);                                     \
		if (rc != EXR_ERR_SUCCESS) {                                  \
			strata_fail(err, "OpenEXR: %s (%s)",                  \
				    exr_get_error_code_as_string(rc), #call); \
			goto out;                                             \
		}                                                             \
	} while (0)

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
	int ret = -1;

	EXR(exr_get_storage(ctx, 0, &storage));
	if (storage != EXR_STORAGE_SCANLINE) {
		strata_fail(err, "input EXR must be a flat scanline image");
		goto out;
	}
	EXR(exr_get_data_window(ctx, 0, window));
	width = (int64_t)window->max.x - window->min.x + 1;
	height = (int64_t)window->max.y - window->min.y + 1;
	if (width > INT_MAX || width < 1 || height > INT_MAX || height < 1) {
		strata_fail(err, "invalid EXR data window");
		goto out;
	}
	if (strata_dimensions((int)width, (int)height, err))
		goto out;
	image->width = (int)width;
	image->height = (int)height;
	image->linear = 1;
	pixels = (size_t)width * (size_t)height;
	image->rgba = calloc(pixels * 4, sizeof(float));
	if (!image->rgba) {
		strata_fail(err, "out of memory");
		goto out;
	}
	for (size_t i = 0; i < pixels; ++i)
		image->rgba[i * 4 + 3] = 1;
	ret = 0;
out:
	return ret;
}

int strata_read_exr(const char *path, struct strata_image *image,
		    struct strata_error *err)
{
	exr_context_t ctx = NULL;
	exr_decode_pipeline_t pipe = { 0 };
	exr_attr_box2i_t window;
	int ret = -1, first = 1, row = 0;

#if OPENEXR_VERSION_MAJOR > 3 || OPENEXR_VERSION_MINOR >= 3
	pipe.pipe_size = sizeof(pipe);
#endif
	EXR(exr_start_read(&ctx, path, NULL));
	if (allocate_exr(ctx, image, &window, err))
		goto out;
	while (row < image->height) {
		exr_chunk_info_t chunk;

		EXR(exr_read_scanline_chunk_info(ctx, 0, window.min.y + row,
						 &chunk));
		if (first)
			EXR(exr_decoding_initialize(ctx, 0, &chunk, &pipe));
		else
			EXR(exr_decoding_update(ctx, 0, &chunk, &pipe));
		if (decode_channels(&pipe, image, row, err))
			goto out;
		EXR(exr_decoding_choose_default_routines(ctx, 0, &pipe));
		EXR(exr_decoding_run(ctx, 0, &pipe));
		first = 0;
		row += chunk.height;
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

static int write_metadata(exr_context_t ctx, int part, int deep,
			  const struct strata_frame_metadata *frame,
			  struct strata_error *err)
{
	const struct strata_camera *camera = &frame->camera;
	char index[32];
	int ret = -1;

	snprintf(index, sizeof(index), "%" PRId64, frame->index);
	EXR(exr_attr_set_string(ctx, part, "strata:frame_index", index));
	EXR(exr_attr_set_double(ctx, part, "strata:timestamp_seconds",
				frame->timestamp_seconds));
	EXR(exr_attr_set_string(ctx, part, "strata:depth_units", "metres"));
	EXR(exr_attr_set_string(ctx, part, "strata:depth_kind", "camera-Z"));
	EXR(exr_attr_set_float(ctx, part, "strata:focal_px", camera->fx_px));
	EXR(exr_attr_set_float(ctx, part, "strata:fx_px", camera->fx_px));
	EXR(exr_attr_set_float(ctx, part, "strata:fy_px", camera->fy_px));
	EXR(exr_attr_set_float(ctx, part, "strata:cx_px", camera->cx_px));
	EXR(exr_attr_set_float(ctx, part, "strata:cy_px", camera->cy_px));
	EXR(exr_attr_set_string(ctx, part, "strata:focal_source",
				camera->focal_source == STRATA_FOCAL_PROVIDED ?
					"provided" :
					"estimated"));
	EXR(exr_attr_set_float(ctx, part, "strata:sensor_width_mm",
			       camera->sensor_width_mm));
	EXR(exr_attr_set_float(ctx, part, "strata:sensor_height_mm",
			       camera->sensor_height_mm));
	EXR(exr_attr_set_int(ctx, part, "strata:sensor_used", 0));
	if (deep) {
		EXR(exr_attr_set_string(ctx, part, "strata:rgb_association",
					"premultiplied"));
		EXR(exr_attr_set_string(ctx, part, "strata:alpha_semantics",
					"coverage"));
		EXR(exr_attr_set_string(ctx, part, "strata:color_space",
					frame->linear ? "scene-linear" :
							"display-sRGB"));
	}
	ret = 0;
out:
	return ret;
}

enum exr_kind { STRATA_EXR_DEPTH, STRATA_EXR_DEEP, STRATA_EXR_POSITION };

static int write_header(exr_context_t ctx, int w, int h, enum exr_kind kind,
			const struct strata_frame_metadata *frame,
			struct strata_error *err)
{
	const char *deep_names[] = { "A", "B", "G", "R", "Z", "ZBack" };
	const char *depth_names[] = { "Z", "network_inverse_depth",
				      "inverse_metres" };
	const char *position_names[] = { "P.x", "P.y", "P.z" };
	int deep = kind == STRATA_EXR_DEEP;
	const char *const *names = deep ? deep_names :
				   kind == STRATA_EXR_POSITION ?
					  position_names :
					  depth_names;
	size_t count = deep ? 6u : 3u;
	int part, ret = -1;

	EXR(exr_add_part(ctx, NULL,
			 deep ? EXR_STORAGE_DEEP_SCANLINE :
				EXR_STORAGE_SCANLINE,
			 &part));
	EXR(exr_initialize_required_attr_simple(ctx, part, w, h,
						deep ? EXR_COMPRESSION_NONE :
						       EXR_COMPRESSION_ZIP));
	if (deep)
		EXR(exr_set_version(ctx, part, 1));
	for (size_t c = 0; c < count; ++c)
		EXR(exr_add_channel(ctx, part, names[c], EXR_PIXEL_FLOAT,
				    EXR_PERCEPTUALLY_LINEAR, 1, 1));
	if (write_metadata(ctx, part, deep, frame, err))
		goto out;
	ret = 0;
out:
	return ret;
}

struct exr_plane {
	const char *name;
	const float *data;
	int stride;
};

static const struct exr_plane *find_plane(const struct exr_plane *planes,
					  const char *name)
{
	for (size_t i = 0; i < 3; ++i) {
		if (!strcmp(name, planes[i].name))
			return &planes[i];
	}
	return NULL;
}

static int write_flat(exr_context_t ctx, int width, int height,
		      const struct exr_plane *planes, struct strata_error *err)
{
	exr_encode_pipeline_t pipe = { 0 };
	int first = 1, row = 0, ret = -1;

	while (row < height) {
		exr_chunk_info_t chunk;

		EXR(exr_write_scanline_chunk_info(ctx, 0, row, &chunk));
		if (first)
			EXR(exr_encoding_initialize(ctx, 0, &chunk, &pipe));
		else
			EXR(exr_encoding_update(ctx, 0, &chunk, &pipe));
		for (int c = 0; c < pipe.channel_count; ++c) {
			exr_coding_channel_info_t *channel = &pipe.channels[c];
			const struct exr_plane *plane =
				find_plane(planes, channel->channel_name);

			if (!plane) {
				strata_fail(err, "unmapped EXR output channel");
				goto out;
			}
			channel->user_data_type = EXR_PIXEL_FLOAT;
			channel->user_bytes_per_element = 4;
			channel->user_pixel_stride = plane->stride * 4;
			channel->user_line_stride = width * plane->stride * 4;
			channel->encode_from_ptr =
				(const uint8_t *)(plane->data +
						  (size_t)row * (size_t)width *
							  (size_t)plane->stride);
		}
		EXR(exr_encoding_choose_default_routines(ctx, 0, &pipe));
		EXR(exr_encoding_run(ctx, 0, &pipe));
		first = 0;
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
		EXR(exr_write_deep_scanline_chunk(
			ctx, 0, row, planes, (uint64_t)24 * samples,
			(uint64_t)24 * samples, counts, (uint64_t)4 * width));
		offset += samples;
	}
	ret = 0;
out:
	free(planes);
	free(counts);
	return ret;
}

static int close_file(exr_context_t *ctx, int ret, struct strata_error *err)
{
	if (*ctx && exr_finish(ctx) != EXR_ERR_SUCCESS && !ret)
		return strata_fail(err, "OpenEXR close failed");
	return ret;
}

int strata_write_depth(const char *path, const struct strata_result *result,
		       struct strata_error *err)
{
	exr_context_t ctx = NULL;
	struct exr_plane planes[3];
	int ret = -1;

	if (!path || !result || !result->camera_z_metres ||
	    !result->network_inverse_depth || !result->inverse_metres)
		return strata_fail(err, "invalid EXR output arguments");
	if (strata_dimensions(result->width, result->height, err))
		return -1;
	EXR(exr_start_write(&ctx, path, EXR_WRITE_FILE_DIRECTLY, NULL));
	if (write_header(ctx, result->width, result->height, STRATA_EXR_DEPTH,
			 &result->frame, err))
		goto out;
	planes[0] = (struct exr_plane){ "Z", result->camera_z_metres, 1 };
	planes[1] = (struct exr_plane){ "network_inverse_depth",
					result->network_inverse_depth, 1 };
	planes[2] = (struct exr_plane){ "inverse_metres",
					result->inverse_metres, 1 };
	EXR(exr_write_header(ctx));
	ret = write_flat(ctx, result->width, result->height, planes, err);
out:
	return close_file(&ctx, ret, err);
}

int strata_write_deep(const char *path, const struct strata_deep_result *deep,
		      struct strata_error *err)
{
	exr_context_t ctx = NULL;
	int ret = -1;

	if (!path)
		return strata_fail(err, "missing EXR output path");
	if (strata_deep_validate(deep, err))
		return -1;
	EXR(exr_start_write(&ctx, path, EXR_WRITE_FILE_DIRECTLY, NULL));
	if (write_header(ctx, deep->width, deep->height, STRATA_EXR_DEEP,
			 &deep->frame, err))
		goto out;
	EXR(exr_write_header(ctx));
	ret = write_deep(ctx, deep, err);
out:
	return close_file(&ctx, ret, err);
}

int strata_write_position(const char *path,
			  const struct strata_positions *positions,
			  struct strata_error *err)
{
	exr_context_t ctx = NULL;
	struct exr_plane planes[3];
	exr_attr_m44f_t matrix;
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
	EXR(exr_start_write(&ctx, path, EXR_WRITE_FILE_DIRECTLY, NULL));
	if (write_header(ctx, positions->width, positions->height,
			 STRATA_EXR_POSITION, &positions->frame, err))
		goto out;
	EXR(exr_attr_set_string(
		ctx, 0, "strata:P_space",
		positions->space == STRATA_SPACE_WORLD ? "world" : "camera"));
	EXR(exr_attr_set_string(ctx, 0, "strata:P_invalid", "zero XYZ"));
	EXR(exr_attr_set_string(ctx, 0, "strata:pose_convention",
				"row-major, column vectors, metres"));
	memcpy(matrix.m, positions->pose.camera_to_world, sizeof(matrix.m));
	EXR(exr_attr_set_m44f(ctx, 0, "strata:camera_to_world", &matrix));
	EXR(exr_write_header(ctx));
	planes[0] = (struct exr_plane){ "P.x", positions->xyz, 3 };
	planes[1] = (struct exr_plane){ "P.y", positions->xyz + 1, 3 };
	planes[2] = (struct exr_plane){ "P.z", positions->xyz + 2, 3 };
	ret = write_flat(ctx, positions->width, positions->height, planes, err);
out:
	return close_file(&ctx, ret, err);
}

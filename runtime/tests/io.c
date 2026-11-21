#include "strata.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define WIDTH 7
#define HEIGHT 19

static int read_inputs(const char *directory, const float *expected)
{
	struct strata_image image = { 0 };
	struct strata_error err = { { 0 } };
	char path[4096];
	const char *names[] = { "input.exr", "input.png", "input.jpg" };
	int ret = -1;

	for (size_t file = 0; file < 3; ++file) {
		snprintf(path, sizeof(path), "%s/%s", directory, names[file]);
		if (strata_image_read(path, &image, &err))
			goto out;
		if (file == 0) {
			if (image.width != WIDTH || image.height != HEIGHT ||
			    !image.linear)
				goto out;
			for (size_t i = 0; i < WIDTH * HEIGHT; ++i) {
				for (size_t c = 0; c < 3; ++c) {
					if (image.rgba[i * 4 + c] !=
					    expected[i * 4 + c])
						goto out;
				}
				if (image.rgba[i * 4 + 3] != 1)
					goto out;
			}
		} else {
			if (image.width != 2 || image.height != 2 ||
			    image.linear)
				goto out;
			for (size_t i = 0; i < 4; ++i) {
				for (size_t c = 0; c < 3; ++c) {
					if (image.rgba[i * 4 + c] !=
					    128.0f / 255.0f)
						goto out;
				}
			}
		}
		strata_image_free(&image);
	}
	snprintf(path, sizeof(path), "%s/missing.exr", directory);
	if (strata_image_read(path, &image, &err) != -1 || image.rgba)
		goto out;
	ret = 0;
out:
	strata_image_free(&image);
	if (ret)
		fprintf(stderr, "input contract failed: %s\n", err.message);
	return ret;
}

static int write_variable(const char *directory)
{
	uint32_t counts[] = { 0, 3, 1, 2, 1, 0 };
	float rgba[28], z[] = { 3, 1, 2, 5, 6, 4, INFINITY };
	float back[] = { 3.5f, 1.5f, 2.5f, 5.5f, 6.5f, 4.5f, INFINITY };
	struct strata_deep_result deep = {
		.width = 3,
		.height = 2,
		.frame = { .index = 42, .camera.fx_px = 1000, .linear = 1 },
		.sample_count = 7,
		.counts = counts,
		.rgba = rgba,
		.z = z,
		.zback = back
	};
	struct strata_error err = { { 0 } };
	char path[4096];

	for (size_t i = 0; i < 7; ++i) {
		float coverage = i == 1 ? .25f : 1;

		for (size_t c = 0; c < 3; ++c)
			rgba[i * 4 + c] = ((float)(i * 4 + c) - 10) * coverage;
		rgba[i * 4 + 3] = coverage;
	}
	snprintf(path, sizeof(path), "%s/variable.exr", directory);
	if (strata_write_deep(path, &deep, &err))
		goto fail;
	memset(counts, 0, sizeof(counts));
	deep.sample_count = 0;
	snprintf(path, sizeof(path), "%s/empty.exr", directory);
	if (strata_write_deep(path, &deep, &err))
		goto fail;
	return 0;
fail:
	fprintf(stderr, "%s\n", err.message);
	return -1;
}

static int write_positions(const char *directory)
{
	float z[] = { 2, 2, 2, 2 };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .frame.camera = { .fx_px = 2,
							 .fy_px = 4,
							 .cx_px = 1,
							 .cy_px = 1 },
				       .camera_z_metres = z };
	struct strata_pose pose = { 0 };
	struct strata_positions positions = { 0 };
	struct strata_error err = { { 0 } };
	char path[4096];
	int ret = -1;

	int has_pose = 0;

	snprintf(path, sizeof(path), "%s/camera.json", directory);
	if (strata_camera_read(path, &depth.frame.camera, &pose, &has_pose,
			       &err) ||
	    !has_pose)
		goto out;
	for (int world = 0; world < 2; ++world) {
		snprintf(path, sizeof(path), "%s/P-%s.exr", directory,
			 world ? "world" : "camera");
		if (strata_unproject_depth(&depth, NULL, world ? &pose : NULL,
					   &positions, &err) ||
		    strata_write_position(path, &positions, &err))
			goto out;
		strata_positions_free(&positions);
	}
	ret = 0;
out:
	if (ret)
		fprintf(stderr, "%s\n", err.message);
	strata_positions_free(&positions);
	return ret;
}

int main(int argc, char **argv)
{
	float rgba[WIDTH * HEIGHT * 4], depth[WIDTH * HEIGHT],
		inverse[WIDTH * HEIGHT], inverse_metres[WIDTH * HEIGHT];
	struct strata_image image = { WIDTH, HEIGHT, 1, rgba };
	struct strata_result result = {
		.width = WIDTH,
		.height = HEIGHT,
		.frame = { .index = 1001,
			   .timestamp_seconds = 1,
			   .camera = { .fx_px = 1000,
				       .fy_px = 1000,
				       .sensor_width_mm = 36,
				       .sensor_height_mm = 24,
				       .focal_source = STRATA_FOCAL_PROVIDED } },
		.network_inverse_depth = inverse,
		.inverse_metres = inverse_metres,
		.camera_z_metres = depth
	};
	struct strata_error err = { { 0 } };
	struct strata_deep_result deep = { 0 };
	char path[4096];

	if (argc != 2)
		return 2;
	for (size_t i = 0; i < WIDTH * HEIGHT * 4; ++i)
		rgba[i] = (float)i / 31 - 1;
	for (size_t i = 0; i < WIDTH * HEIGHT; ++i)
		depth[i] = (float)i + .25f;
	depth[2 * WIDTH + 3] = INFINITY;
	for (size_t i = 0; i < WIDTH * HEIGHT; ++i)
		inverse[i] = 1000 / (WIDTH * depth[i]);
	for (size_t i = 0; i < WIDTH * HEIGHT; ++i)
		inverse_metres[i] = 1 / depth[i];
	if (read_inputs(argv[1], rgba) || write_variable(argv[1]) ||
	    write_positions(argv[1]))
		return 1;
	snprintf(path, sizeof(path), "%s/depth.exr", argv[1]);
	if (strata_write_depth(path, &result, &err))
		goto fail;
	snprintf(path, sizeof(path), "%s/deep.exr", argv[1]);
	if (strata_depth_to_deep(&image, &result, &deep, &err) ||
	    strata_write_deep(path, &deep, &err))
		goto fail;
	if (strata_write_depth(path, NULL, &err) != -1 ||
	    strata_write_deep(path, NULL, &err) != -1)
		goto fail;
	strata_deep_free(&deep);
	return 0;
fail:
	strata_deep_free(&deep);
	fprintf(stderr, "%s\n", err.message);
	return 1;
}

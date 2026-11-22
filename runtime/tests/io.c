#define _POSIX_C_SOURCE 200809L
#include "strata.h"
#include <dirent.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include "expect.h"

#define WIDTH 7
#define HEIGHT 19

static int truncated(const char *directory)
{
	char path[4096], bytes[65536];
	FILE *file;
	size_t count;

	snprintf(path, sizeof(path), "%s/input.exr", directory);
	file = fopen(path, "rb");
	if (!file)
		return -1;
	count = fread(bytes, 1, sizeof(bytes), file);
	fclose(file);
	snprintf(path, sizeof(path), "%s/truncated.exr", directory);
	file = fopen(path, "wb");
	if (!file)
		return -1;
	count = fwrite(bytes, 1, count - 16, file);
	return fclose(file) || !count ? -1 : 0;
}

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
	}
	snprintf(path, sizeof(path), "%s/missing.exr", directory);
	if (strata_image_read(path, &image, &err) != -1 ||
	    !zeroed(&image, sizeof(image)))
		goto out;
	snprintf(path, sizeof(path), "%s/%s", directory, names[0]);
	if (strata_image_read(path, &image, &err) || truncated(directory))
		goto out;
	snprintf(path, sizeof(path), "%s/truncated.exr", directory);
	if (strata_image_read(path, &image, &err) != -1 ||
	    !zeroed(&image, sizeof(image)))
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
	float z[] = { .5f, .5f, .5f, .5f };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .network_inverse_depth = z };
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

static int rejected_camera(const char *directory)
{
	struct strata_camera camera = { .fx_px = 7 };
	struct strata_pose pose = { .camera_to_world = { 7 } };
	struct strata_error err = { { 0 } };
	char path[4096];
	FILE *file;
	int has_pose = 7;

	snprintf(path, sizeof(path), "%s/rejected-camera.json", directory);
	file = fopen(path, "wb");
	if (!file)
		return -1;
	fputs("{\"version\": 1, \"fx_px\": 100, \"fy_px\": 100, \"cx_px\": 1, "
	      "\"cy_px\": 1, \"camera_to_world\": [1, 0, 0, 0, 0, 1, 0, 0, "
	      "0, 0, 1, 0, 0, 0, 0, 1], \"unknown\": 0}",
	      file);
	if (fclose(file))
		return -1;
	if (strata_camera_read(path, &camera, &pose, &has_pose, &err) != -1 ||
	    !strstr(err.message, "unknown camera JSON field") ||
	    !zeroed(&camera, sizeof(camera)) || !zeroed(&pose, sizeof(pose)) ||
	    has_pose) {
		fprintf(stderr, "rejected camera left outputs filled\n");
		return -1;
	}
	return 0;
}

static int read_text(const char *path, char *text, size_t size)
{
	FILE *file = fopen(path, "rb");
	size_t count;

	if (!file)
		return -1;
	count = fread(text, 1, size - 1, file);
	text[count] = '\0';
	fclose(file);
	return (int)count;
}

static int leftovers(const char *directory, const char *name)
{
	DIR *dir = opendir(directory);
	struct dirent *entry;
	size_t length = strlen(name);
	int found = 0;

	if (!dir)
		return -1;
	while ((entry = readdir(dir)))
		found += !strncmp(entry->d_name, name, length) &&
			 entry->d_name[length] == '.';
	closedir(dir);
	return found;
}

static int limited_write(const char *path, const struct strata_result *result,
			 struct strata_error *err)
{
	struct rlimit original, limited;
	int ret;

	if (getrlimit(RLIMIT_FSIZE, &original))
		return 0;
	limited = original;
	limited.rlim_cur = 65536;
	signal(SIGXFSZ, SIG_IGN);
	if (setrlimit(RLIMIT_FSIZE, &limited))
		return 0;
	ret = strata_write_depth(path, result, err);
	setrlimit(RLIMIT_FSIZE, &original);
	return ret;
}

static int atomic_test(const char *directory)
{
	const size_t side = 512;
	float *q = malloc(side * side * sizeof(float));
	struct strata_result result = {
		.width = (int)side,
		.height = (int)side,
		.frame.camera = { .fx_px = 1000,
				  .fy_px = 1000,
				  .focal_source = STRATA_FOCAL_PROVIDED },
		.network_inverse_depth = q
	};
	struct strata_error err = { { 0 } };
	char path[4096], text[64];
	FILE *file;
	int ret = -1;

	snprintf(path, sizeof(path), "%s/atomic.exr", directory);
	file = fopen(path, "wb");
	if (!q || !file)
		goto out;
	fputs("keep\n", file);
	fclose(file);
	srand(7);
	for (size_t i = 0; i < side * side; ++i)
		q[i] = (float)rand() / (float)RAND_MAX + .01f;
	if (limited_write(path, &result, &err) != -1 ||
	    read_text(path, text, sizeof(text)) != 5 ||
	    strcmp(text, "keep\n") || leftovers(directory, "atomic.exr") != 0) {
		fprintf(stderr, "failed write replaced or left output\n");
		goto out;
	}
	if (strata_write_depth(path, &result, &err) ||
	    read_text(path, text, sizeof(text)) <= 5 ||
	    leftovers(directory, "atomic.exr") != 0) {
		fprintf(stderr, "atomic replacement failed: %s\n", err.message);
		goto out;
	}
	ret = 0;
out:
	free(q);
	return ret;
}

int main(int argc, char **argv)
{
	float rgba[WIDTH * HEIGHT * 4], depth[WIDTH * HEIGHT],
		inverse[WIDTH * HEIGHT];
	struct strata_image image = {
		.width = WIDTH, .height = HEIGHT, .linear = 1, .rgba = rgba
	};
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
		.network_inverse_depth = inverse
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
	if (read_inputs(argv[1], rgba) || write_variable(argv[1]) ||
	    write_positions(argv[1]) || rejected_camera(argv[1]) ||
	    atomic_test(argv[1]))
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

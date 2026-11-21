#define _POSIX_C_SOURCE 200809L
#include "strata.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <errno.h>

struct cli_options {
	struct strata_options session;
	const char *input;
	const char *depth;
	const char *deep;
	const char *position;
	const char *camera;
	float focal;
	float exposure;
	int repeat;
};

static double milliseconds(void)
{
	struct timespec time;

#ifdef _WIN32
	timespec_get(&time, TIME_UTC);
#else
	clock_gettime(CLOCK_MONOTONIC, &time);
#endif
	return 1000.0 * (double)time.tv_sec + (double)time.tv_nsec / 1000000.0;
}

static void usage(FILE *out)
{
	fputs("strata depth --model M.onnx --in plate.exr|png|jpg --out-depth depth.exr\n"
	      "             --out-deep deep.exr [--f-px N] [--exposure stops]\n"
	      "             [--provider auto|cuda|tensorrt|coreml|directml|cpu]\n"
	      "             [--device N] [--repeat N] [--out-position P.exr] [--camera camera.json]\n"
	      "No --f-px: estimate focal using M.fov.onnx. --repeat: time N frames.\n",
	      out);
}

static int numeric_option(struct cli_options *options, const char *key,
			  const char *value)
{
	char *end;
	double number;

	errno = 0;
	number = strtod(value, &end);
	if (errno || end == value || *end || !isfinite(number))
		return -1;
	if (!strcmp(key, "--f-px") && number > 0 && number <= 1e9)
		options->focal = (float)number;
	else if (!strcmp(key, "--exposure") && fabs(number) <= 100)
		options->exposure = (float)number;
	else if (!strcmp(key, "--repeat") && number >= 1 && number <= 10000 &&
		 number == floor(number))
		options->repeat = (int)number;
	else if (!strcmp(key, "--device") && number >= 0 && number <= 1024 &&
		 number == floor(number))
		options->session.device_id = (int)number;
	else
		return -1;
	return 0;
}

static int valid_paths(const struct cli_options *options)
{
	const char *inputs[] = { options->input, options->session.model_path,
				 options->camera };
	const char *outputs[] = { options->depth, options->deep,
				  options->position };

	for (size_t i = 0; i < 3; ++i) {
		if (!outputs[i])
			continue;
		for (size_t j = 0; j < 3; ++j) {
			if (inputs[j] && !strcmp(outputs[i], inputs[j]))
				return 0;
		}
		for (size_t j = 0; j < i; ++j) {
			if (outputs[j] && !strcmp(outputs[i], outputs[j]))
				return 0;
		}
	}
	return 1;
}

static int parse(int argc, char **argv, struct cli_options *options)
{
	for (int i = 2; i < argc; i += 2) {
		const char *key = argv[i];
		const char *value;

		if (i + 1 == argc)
			return -1;
		value = argv[i + 1];
		if (!strcmp(key, "--model"))
			options->session.model_path = value;
		else if (!strcmp(key, "--in"))
			options->input = value;
		else if (!strcmp(key, "--out-depth"))
			options->depth = value;
		else if (!strcmp(key, "--out-deep"))
			options->deep = value;
		else if (!strcmp(key, "--out-position"))
			options->position = value;
		else if (!strcmp(key, "--camera"))
			options->camera = value;
		else if (!strcmp(key, "--provider"))
			options->session.provider = value;
		else if (numeric_option(options, key, value))
			return -1;
	}
	if (!options->session.model_path || !options->input ||
	    !options->depth || !options->deep)
		return -1;
	if (!valid_paths(options) || (options->camera && options->focal > 0))
		return -1;
	return 0;
}

static void report(strata_context *ctx, const struct strata_image *image,
		   const struct strata_result *result, double session_ms,
		   double write_ms, const double *times, int repeat)
{
	printf("{\"provider\":\"%s\",\"width\":%d,\"height\":%d,\"focal_px\":%.9g,"
	       "\"session_ms\":%.3f,\"write_ms\":%.3f,\"frame_ms\":[",
	       strata_provider(ctx), image->width, image->height,
	       result->frame.camera.fx_px, session_ms, write_ms);
	for (int i = 0; i < repeat; ++i)
		printf("%s%.3f", i ? "," : "", times[i]);
	puts("]}");
}

static int write_position(const struct cli_options *options,
			  const struct strata_deep_result *deep,
			  const struct strata_pose *pose,
			  struct strata_error *err)
{
	struct strata_layers layers = { 0 };
	struct strata_positions positions = { 0 };
	int ret = -1;

	if (!options->position)
		return 0;
	if (strata_deep_to_layers(deep, 1, &layers, err) ||
	    strata_unproject_layers(&layers, NULL, pose, &positions, err) ||
	    strata_write_position(options->position, &positions, err))
		goto out;
	ret = 0;
out:
	strata_positions_free(&positions);
	strata_layers_free(&layers);
	return ret;
}

static int execute(const struct cli_options *options)
{
	struct strata_error err = { { 0 } };
	struct strata_image image = { 0 };
	struct strata_result result = { 0 };
	struct strata_deep_result deep = { 0 };
	struct strata_frame frame = { .image = &image,
				      .exposure_stops = options->exposure };
	strata_context *ctx = NULL;
	struct strata_pose pose = { 0 };
	int has_pose = 0;
	double *times = calloc((size_t)options->repeat, sizeof(*times));
	double start, session_ms;
	int ret = 1;

	if (!times) {
		fputs("out of memory\n", stderr);
		return 1;
	}
	if (strata_image_read(options->input, &image, &err))
		goto out;
	if (options->focal > 0)
		frame.camera =
			(struct strata_camera){ .fx_px = options->focal,
						.fy_px = options->focal,
						.cx_px = (float)image.width / 2,
						.cy_px = (float)image.height /
							 2 };
	if (options->camera &&
	    strata_camera_read(options->camera, &frame.camera, &pose, &has_pose,
			       &err))
		goto out;
	start = milliseconds();
	if (strata_open(&options->session, &ctx, &err))
		goto out;
	session_ms = milliseconds() - start;
	for (int i = 0; i < options->repeat; ++i) {
		strata_result_free(&result);
		start = milliseconds();
		if (strata_run(ctx, &frame, 1, &result, &err))
			goto out;
		times[i] = milliseconds() - start;
	}
	start = milliseconds();
	if (strata_depth_to_deep(&image, &result, &deep, &err) ||
	    strata_write_depth(options->depth, &result, &err) ||
	    strata_write_deep(options->deep, &deep, &err) ||
	    write_position(options, &deep, has_pose ? &pose : NULL, &err))
		goto out;
	report(ctx, &image, &result, session_ms, milliseconds() - start, times,
	       options->repeat);
	ret = 0;
out:
	if (ret)
		fprintf(stderr, "strata: %s\n", err.message);
	strata_deep_free(&deep);
	strata_result_free(&result);
	strata_image_free(&image);
	strata_close(ctx);
	free(times);
	return ret;
}

int main(int argc, char **argv)
{
	struct cli_options options = { .session.provider = "auto",
				       .repeat = 1 };

	if ((argc == 2 &&
	     (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) ||
	    (argc == 3 && !strcmp(argv[1], "depth") &&
	     !strcmp(argv[2], "--help"))) {
		usage(stdout);
		return 0;
	}
	if (argc < 2 || strcmp(argv[1], "depth") ||
	    parse(argc, argv, &options)) {
		usage(stderr);
		return 2;
	}
	return execute(&options);
}

#define _XOPEN_SOURCE 700
#include "strata.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct cli_options {
	struct strata_options session;
	const char *head;
	const char *input;
	const char *depth;
	const char *deep;
	const char *position;
	const char *camera;
	float focal;
	double exposure;
};

static double milliseconds(void)
{
	struct timespec time;

	clock_gettime(CLOCK_MONOTONIC, &time);
	return 1000.0 * (double)time.tv_sec + (double)time.tv_nsec / 1000000.0;
}

static void usage(FILE *out)
{
	fputs("strata depth --model M.onnx --in plate.exr|png|jpg\n"
	      "             --out-depth depth.exr [--head H.onnx --out-deep deep.exr]\n"
	      "             [--f-px N | --camera camera.json] [--exposure stops]\n"
	      "             [--provider auto|cuda|cpu] [--device N]\n"
	      "             [--out-position P.exr]\n"
	      "Without --f-px or camera intrinsics, M.fov.onnx estimates focal.\n",
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
		options->exposure = number;
	else if (!strcmp(key, "--device") && number >= 0 && number <= 1024 &&
		 number == floor(number))
		options->session.device_id = (int)number;
	else
		return -1;
	return 0;
}

static int clashes(const char *output, const char *const *paths, size_t count)
{
	struct strata_error err;
	int same = 0;

	for (size_t i = 0; i < count; ++i) {
		if (paths[i] &&
		    (strata_same_file(output, paths[i], &same, &err) || same))
			return 1;
	}
	return 0;
}

static char *resolved_parent(const char *path)
{
	char *copy = strdup(path), *slash, *parent;

	if (!copy)
		return NULL;
	slash = strrchr(copy, '/');
	if (slash)
		slash[slash == copy] = '\0';
	parent = realpath(slash ? copy : ".", NULL);
	free(copy);
	return parent;
}

static int inside(const char *directory, const char *path)
{
	size_t length = strlen(directory);

	return !strncmp(directory, path, length) &&
	       (path[length] == '/' || !path[length] || length == 1);
}

static int valid_paths(const struct cli_options *options,
		       const char *const *model, const char *const *directories)
{
	const char *const paths[] = { options->depth, options->deep,
				      options->position, options->input,
				      options->camera };

	for (size_t i = 0; i < 3; ++i) {
		char *parent;
		int refused;

		if (!paths[i])
			continue;
		parent = resolved_parent(paths[i]);
		refused = !parent || inside(directories[0], parent) ||
			  (directories[1] && inside(directories[1], parent)) ||
			  clashes(paths[i], paths + i + 1, 4 - i) ||
			  clashes(paths[i], model, 3);
		free(parent);
		if (refused)
			return 0;
	}
	return 1;
}

static int check_model_paths(const struct cli_options *options,
			     const char *path)
{
	char *manifest = strata_manifest_path(path);
	char *fov = strata_fov_path(path);
	char *real = realpath(path, NULL);
	char *directory = resolved_parent(path);
	char *real_directory = real ? resolved_parent(real) : NULL;
	const char *const model[] = { path, manifest, fov };
	const char *const directories[] = { directory, real_directory };
	int ret = 0;

	if (!manifest || !fov || !directory) {
		fprintf(stderr, "strata: cannot resolve model path: %s\n",
			path);
		ret = 1;
	} else if (!valid_paths(options, model, directories)) {
		usage(stderr);
		ret = 2;
	}
	free(manifest);
	free(fov);
	free(real);
	free(directory);
	free(real_directory);
	return ret;
}

static int check_paths(const struct cli_options *options)
{
	int ret = check_model_paths(options, options->session.model_path);

	return ret || !options->head ?
		       ret :
		       check_model_paths(options, options->head);
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
		else if (!strcmp(key, "--head"))
			options->head = value;
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
	    !options->depth || (!!options->deep != !!options->head))
		return -1;
	if (options->camera && options->focal > 0)
		return -1;
	return 0;
}

static void json_string(const char *text)
{
	putchar('"');
	for (const unsigned char *c = (const unsigned char *)text; *c; ++c) {
		if (*c == '"' || *c == '\\')
			printf("\\%c", *c);
		else if (*c < 0x20)
			printf("\\u%04x", *c);
		else
			putchar(*c);
	}
	putchar('"');
}

static void report(strata_context *ctx, const struct strata_result *result,
		   double session_ms, double frame_ms, double write_ms)
{
	const struct strata_timings *t = &result->timings;

	printf("{\"provider\":\"%s\",\"fallback_reason\":",
	       strata_provider(ctx));
	json_string(strata_fallback_reason(ctx));
	printf(",\"width\":%d,\"height\":%d,\"focal_px\":%.9g,"
	       "\"focal_source\":\"%s\",\"nonfinite_input_pixels\":%zu,"
	       "\"session_ms\":%.3f,\"frame_ms\":%.3f,\"write_ms\":%.3f,"
	       "\"stages\":{\"prepare_ms\":%.3f,\"focal_ms\":%.3f,"
	       "\"upload_ms\":%.3f,\"run_ms\":%.3f,\"download_ms\":%.3f,"
	       "\"postprocess_ms\":%.3f}}\n",
	       result->width, result->height, result->frame.camera.fx_px,
	       result->frame.camera.focal_source == STRATA_FOCAL_PROVIDED ?
		       "provided" :
		       "estimated",
	       result->nonfinite_input_pixels, session_ms, frame_ms, write_ms,
	       t->prepare_ms, t->focal_ms, t->upload_ms, t->run_ms,
	       t->download_ms, t->postprocess_ms);
}

static int write_outputs(const struct cli_options *options,
			 const struct strata_deep_result *deep,
			 const struct strata_result *result,
			 const struct strata_pose *pose,
			 struct strata_error *err)
{
	struct strata_positions positions = { 0 };
	int ret = -1;

	if (strata_write_depth(options->depth, result, err) ||
	    (options->deep && strata_write_deep(options->deep, deep, err)))
		goto out;
	if (options->position &&
	    (strata_unproject_depth(result, NULL, pose, &positions, err) ||
	     strata_write_position(options->position, &positions, err)))
		goto out;
	ret = 0;
out:
	strata_positions_free(&positions);
	return ret;
}

static int frame_camera(const struct cli_options *options,
			const struct strata_image *image,
			struct strata_frame *frame, struct strata_pose *pose,
			int *has_pose, struct strata_error *err)
{
	frame->camera.focal_source = STRATA_FOCAL_ESTIMATED;
	if (options->camera)
		return strata_camera_read(options->camera, &frame->camera, pose,
					  has_pose, err);
	if (options->focal > 0)
		frame->camera = (struct strata_camera){
			.fx_px = options->focal,
			.fy_px = options->focal,
			.cx_px = (float)image->width / 2,
			.cy_px = (float)image->height / 2,
			.focal_source = STRATA_FOCAL_PROVIDED
		};
	return 0;
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
	int has_pose = 0, ret = 1;
	double start, session_ms, frame_ms;

	if (strata_image_read(options->input, &image, &err) ||
	    frame_camera(options, &image, &frame, &pose, &has_pose, &err))
		goto out;
	start = milliseconds();
	if (strata_open(&options->session, &ctx, &err) ||
	    (options->head && strata_set_head(ctx, options->head, &err)))
		goto out;
	session_ms = milliseconds() - start;
	start = milliseconds();
	if (options->head ? strata_run_deep(ctx, &frame, &result, &deep, &err) :
			    strata_run(ctx, &frame, &result, &err))
		goto out;
	frame_ms = milliseconds() - start;
	start = milliseconds();
	if (write_outputs(options, &deep, &result, has_pose ? &pose : NULL,
			  &err))
		goto out;
	report(ctx, &result, session_ms, frame_ms, milliseconds() - start);
	ret = 0;
out:
	if (ret)
		fprintf(stderr, "strata: %s\n", err.message);
	strata_result_free(&result);
	strata_deep_free(&deep);
	strata_image_free(&image);
	strata_close(ctx);
	return ret;
}

int main(int argc, char **argv)
{
	struct cli_options options = { .session.provider = "auto" };
	int ret;

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
	ret = check_paths(&options);
	return ret ? ret : execute(&options);
}

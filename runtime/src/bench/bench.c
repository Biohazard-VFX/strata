#define _POSIX_C_SOURCE 200809L
#include "strata.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct bench_options {
	struct strata_options session;
	const char *input;
	float focal;
	double exposure;
	long frames;
};

static double milliseconds(void)
{
	struct timespec time;

	clock_gettime(CLOCK_MONOTONIC, &time);
	return 1000.0 * (double)time.tv_sec + (double)time.tv_nsec / 1000000.0;
}

static int parse(int argc, char **argv, struct bench_options *options)
{
	for (int i = 1; i + 1 < argc; i += 2) {
		const char *key = argv[i], *value = argv[i + 1];
		char *end;
		double number = strtod(value, &end);
		int numeric = end != value && !*end && isfinite(number);

		if (!strcmp(key, "--model"))
			options->session.model_path = value;
		else if (!strcmp(key, "--in"))
			options->input = value;
		else if (!strcmp(key, "--provider"))
			options->session.provider = value;
		else if (!strcmp(key, "--device") && numeric && number >= 0)
			options->session.device_id = (int)number;
		else if (!strcmp(key, "--f-px") && numeric && number > 0)
			options->focal = (float)number;
		else if (!strcmp(key, "--exposure") && numeric)
			options->exposure = number;
		else if (!strcmp(key, "--frames") && numeric && number >= 1 &&
			 number <= 100000 && number == floor(number))
			options->frames = (long)number;
		else
			return -1;
	}
	return argc % 2 == 1 && options->session.model_path && options->input &&
			       options->frames ?
		       0 :
		       -1;
}

static void print_frame(long index, double frame_ms,
			const struct strata_result *result)
{
	const struct strata_timings *t = &result->timings;

	printf("%s{\"frame_ms\":%.3f,\"prepare_ms\":%.3f,\"focal_ms\":%.3f,"
	       "\"upload_ms\":%.3f,\"run_ms\":%.3f,\"download_ms\":%.3f,"
	       "\"postprocess_ms\":%.3f}",
	       index ? "," : "", frame_ms, t->prepare_ms, t->focal_ms,
	       t->upload_ms, t->run_ms, t->download_ms, t->postprocess_ms);
}

static int frames(strata_context *ctx, const struct bench_options *options,
		  const struct strata_frame *frame, struct strata_error *err)
{
	for (long i = 0; i < options->frames; ++i) {
		struct strata_result result = { 0 };
		double start = milliseconds();

		if (strata_run(ctx, frame, &result, err))
			return -1;
		print_frame(i, milliseconds() - start, &result);
		strata_result_free(&result);
	}
	return 0;
}

static int bench(const struct bench_options *options)
{
	struct strata_error err = { { 0 } };
	struct strata_image image = { 0 };
	struct strata_frame frame = { .image = &image,
				      .exposure_stops = options->exposure };
	strata_context *ctx = NULL;
	double start;
	int ret = 1;

	if (strata_image_read(options->input, &image, &err))
		goto out;
	frame.camera.focal_source = STRATA_FOCAL_ESTIMATED;
	if (options->focal > 0)
		frame.camera = (struct strata_camera){
			.fx_px = options->focal,
			.fy_px = options->focal,
			.cx_px = (float)image.width / 2,
			.cy_px = (float)image.height / 2,
			.focal_source = STRATA_FOCAL_PROVIDED
		};
	start = milliseconds();
	if (strata_open(&options->session, &ctx, &err))
		goto out;
	printf("{\"provider\":\"%s\",\"width\":%d,\"height\":%d,"
	       "\"session_ms\":%.3f,\"frames\":[",
	       strata_provider(ctx), image.width, image.height,
	       milliseconds() - start);
	if (frames(ctx, options, &frame, &err))
		goto out;
	puts("]}");
	ret = 0;
out:
	if (ret)
		fprintf(stderr, "strata_bench: %s\n", err.message);
	strata_image_free(&image);
	strata_close(ctx);
	return ret;
}

int main(int argc, char **argv)
{
	struct bench_options options = { .session.provider = "auto" };

	if (parse(argc, argv, &options)) {
		fputs("strata_bench --model M.onnx --in image --frames N\n"
		      "             [--f-px N] [--exposure stops]\n"
		      "             [--provider auto|cuda|cpu] [--device N]\n",
		      stderr);
		return 2;
	}
	return bench(&options);
}

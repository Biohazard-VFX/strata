#include "strata.h"
#include "expect.h"
#include <math.h>
#include <pthread.h>
#include <string.h>

static float rgba[16];
static const struct strata_image image = { .width = 2,
					   .height = 2,
					   .rgba = rgba };

static void open_failures(const char *bad, const char *missing)
{
	struct strata_options options = { .provider = "cpu" };
	struct strata_error err = { { 0 } };
	strata_context *ctx = (strata_context *)&options;

	options.model_path = missing;
	EXPECT(strata_open(&options, &ctx, &err) == -1 && !ctx);
	EXPECT(strstr(err.message, "model contract missing") != NULL);
	options.model_path = bad;
	EXPECT(strata_open(&options, &ctx, &err) == -1 && !ctx);
	EXPECT(strstr(err.message, "ONNX tensor contract mismatch") != NULL);
	options.provider = "tensorrt";
	EXPECT(strata_open(&options, &ctx, &err) == -1 && !ctx);
	EXPECT(strstr(err.message, "unknown provider") != NULL);
}

static void fallback(const char *model)
{
	struct strata_options options = { .model_path = model,
					  .provider = "cuda",
					  .device_id = 1023 };
	struct strata_error err = { { 0 } };
	strata_context *ctx = NULL;

	EXPECT(strata_open(&options, &ctx, &err) == -1 && !ctx);
	EXPECT(err.message[0] != '\0');
	options.provider = "auto";
	if (EXPECT(!strata_open(&options, &ctx, &err))) {
		EXPECT(!strcmp(strata_provider(ctx), "cpu"));
		EXPECT(strata_fallback_reason(ctx)[0] != '\0');
	}
	strata_close(ctx);
}

static void provided(strata_context *ctx, float *network)
{
	struct strata_frame frame = { .image = &image,
				      .index = 1001,
				      .timestamp_seconds = 1.25,
				      .camera = { .fx_px = 100,
						  .fy_px = 120,
						  .cx_px = .25f,
						  .cy_px = .75f,
						  .sensor_width_mm = 36,
						  .sensor_height_mm = 24 } };
	struct strata_image wide = image;
	struct strata_result result = { 0 };
	struct strata_error err = { { 0 } };
	struct strata_camera *camera = &result.frame.camera;

	EXPECT(strata_run(ctx, &frame, &result, &err) == -1);
	EXPECT(strstr(err.message, "focal source") != NULL);
	frame.camera.focal_source = STRATA_FOCAL_PROVIDED;
	wide.width = 8000;
	if (!EXPECT(!strata_run(ctx, &frame, &result, &err)))
		return;
	*network = result.network_inverse_depth[0];
	EXPECT(result.width == 2 && result.height == 2 && *network > 0);
	EXPECT(strata_inverse_metres(&result, 0) == 2 * *network / 100);
	EXPECT(strata_camera_z_metres(&result, 0) == 100 / (2 * *network));
	EXPECT(result.frame.index == 1001 &&
	       result.frame.timestamp_seconds == 1.25);
	EXPECT(camera->fx_px == 100 && camera->fy_px == 120 &&
	       camera->cx_px == .25f && camera->cy_px == .75f);
	EXPECT(camera->sensor_width_mm == 36 && camera->sensor_height_mm == 24);
	EXPECT(camera->focal_source == STRATA_FOCAL_PROVIDED);
	EXPECT(result.nonfinite_input_pixels == 0);
	frame.camera.fx_px = 200;
	if (EXPECT(!strata_run(ctx, &frame, &result, &err)))
		EXPECT(camera->fx_px == 200 &&
		       result.network_inverse_depth[0] == *network);
	frame.camera.focal_source = STRATA_FOCAL_UNSET;
	EXPECT(strata_run(ctx, &frame, &result, &err) == -1);
	EXPECT_ZERO(result);
	frame.camera.focal_source = STRATA_FOCAL_PROVIDED;
	EXPECT(!strata_run(ctx, &frame, &result, &err));
	frame.image = &wide;
	EXPECT(strata_run(ctx, &frame, &result, &err) == -1);
	EXPECT(strstr(err.message, "dimensions") != NULL);
	EXPECT_ZERO(result);
	frame.image = &image;
	EXPECT(!strata_run(ctx, &frame, &result, &err));
	EXPECT(strata_run(ctx, NULL, &result, &err) == -1);
	EXPECT_ZERO(result);
	strata_result_free(&result);
}

static void estimated(strata_context *ctx)
{
	struct strata_frame frame = {
		.image = &image, .camera.focal_source = STRATA_FOCAL_ESTIMATED
	};
	struct strata_result result = { 0 };
	struct strata_error err = { { 0 } };
	float focal = 1 / tanf(30 * .008726646259971648f);

	frame.camera.cx_px = 1;
	EXPECT(strata_run(ctx, &frame, &result, &err) == -1);
	EXPECT_ZERO(result);
	frame.camera.cx_px = 0;
	if (EXPECT(!strata_run(ctx, &frame, &result, &err))) {
		EXPECT(result.frame.camera.focal_source ==
		       STRATA_FOCAL_ESTIMATED);
		EXPECT(result.frame.camera.fx_px == focal &&
		       result.frame.camera.fy_px == focal);
		EXPECT(result.frame.camera.cx_px == 1 &&
		       result.frame.camera.cy_px == 1);
		EXPECT(result.timings.focal_ms > 0);
	}
	strata_result_free(&result);
}

static void nonfinite(strata_context *ctx)
{
	float dirty[16], clean[16];
	struct strata_image a = image, b = image;
	struct strata_frame frame = {
		.image = &a,
		.camera = { .fx_px = 10,
			    .fy_px = 10,
			    .focal_source = STRATA_FOCAL_PROVIDED }
	};
	struct strata_result first = { 0 }, second = { 0 };
	struct strata_error err = { { 0 } };

	memcpy(dirty, rgba, sizeof(dirty));
	memcpy(clean, rgba, sizeof(clean));
	dirty[0] = INFINITY;
	clean[0] = 1;
	dirty[5] = NAN;
	clean[5] = 0;
	dirty[6] = -INFINITY;
	clean[6] = 0;
	a.rgba = dirty;
	b.rgba = clean;
	EXPECT(!strata_run(ctx, &frame, &first, &err));
	frame.image = &b;
	EXPECT(!strata_run(ctx, &frame, &second, &err));
	EXPECT(first.nonfinite_input_pixels == 2);
	EXPECT(second.nonfinite_input_pixels == 0);
	EXPECT(first.network_inverse_depth && second.network_inverse_depth &&
	       !memcmp(first.network_inverse_depth,
		       second.network_inverse_depth, 4 * sizeof(float)));
	strata_result_free(&first);
	strata_result_free(&second);
}

static void reload(strata_context *ctx, float network)
{
	struct strata_frame frame = {
		.image = &image,
		.camera = { .fx_px = 100,
			    .fy_px = 120,
			    .focal_source = STRATA_FOCAL_PROVIDED }
	};
	struct strata_result result = { 0 };
	struct strata_error err = { { 0 } };

	strata_unload(ctx);
	strata_unload(ctx);
	if (EXPECT(!strata_run(ctx, &frame, &result, &err)))
		EXPECT(result.network_inverse_depth[0] == network);
	strata_result_free(&result);
}

struct worker {
	const char *model;
	float network;
	int ret;
};

static void *work(void *argument)
{
	struct worker *worker = argument;
	struct strata_options options = { .model_path = worker->model,
					  .provider = "cpu" };
	struct strata_frame frame = {
		.image = &image,
		.camera = { .fx_px = 100,
			    .fy_px = 120,
			    .focal_source = STRATA_FOCAL_PROVIDED }
	};
	struct strata_error err = { { 0 } };
	strata_context *ctx = NULL;

	worker->ret = strata_open(&options, &ctx, &err);
	for (int i = 0; !worker->ret && i < 20; ++i) {
		struct strata_result result = { 0 };

		worker->ret = strata_run(ctx, &frame, &result, &err);
		if (!worker->ret &&
		    result.network_inverse_depth[0] != worker->network)
			worker->ret = -1;
		strata_result_free(&result);
	}
	strata_close(ctx);
	return NULL;
}

static void concurrent(const char *model, float network)
{
	struct worker workers[2] = { { model, network, -1 },
				     { model, network, -1 } };
	pthread_t threads[2];

	for (size_t i = 0; i < 2; ++i)
		EXPECT(!pthread_create(&threads[i], NULL, work, &workers[i]));
	for (size_t i = 0; i < 2; ++i) {
		EXPECT(!pthread_join(threads[i], NULL));
		EXPECT(workers[i].ret == 0);
	}
}

int main(int argc, char **argv)
{
	struct strata_options options = { .provider = "cpu" };
	struct strata_error err = { { 0 } };
	strata_context *ctx = NULL;
	float network = 0;

	if (argc != 4)
		return 2;
	for (size_t i = 0; i < 16; ++i)
		rgba[i] = (float)(i % 4 + 1) / 8;
	open_failures(argv[2], argv[3]);
	fallback(argv[1]);
	options.model_path = argv[1];
	if (!EXPECT(!strata_open(&options, &ctx, &err))) {
		fprintf(stderr, "%s\n", err.message);
		return 1;
	}
	EXPECT(!strcmp(strata_provider(ctx), "cpu"));
	EXPECT(!strcmp(strata_fallback_reason(ctx), ""));
	provided(ctx, &network);
	estimated(ctx);
	nonfinite(ctx);
	reload(ctx, network);
	strata_close(ctx);
	concurrent(argv[1], network);
	strata_close(NULL);
	if (!failures)
		puts("session open failures, fallback, focal sources, "
		     "non-finite input, reload and concurrent contexts passed");
	return failures != 0;
}

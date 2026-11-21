#include "strata.h"
#include "expect.h"
#include <math.h>
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
	struct strata_result result = { 0 };
	struct strata_error err = { { 0 } };
	struct strata_camera *camera = &result.frame.camera;

	EXPECT(strata_run(ctx, &frame, &result, &err) == -1);
	EXPECT(strstr(err.message, "focal source") != NULL);
	frame.camera.focal_source = STRATA_FOCAL_PROVIDED;
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
	strata_close(NULL);
	if (!failures)
		puts("session open failures, focal sources, non-finite input "
		     "and reload passed");
	return failures != 0;
}

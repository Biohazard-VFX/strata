#include "strata.h"
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                   \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			goto out;                                          \
		}                                                          \
	} while (0)

int main(int argc, char **argv)
{
	struct strata_image image = { 0 };
	struct strata_result result = { 0 };
	struct strata_options options = { .provider = "cpu" };
	struct strata_frame frame = { .image = &image,
				      .index = 1001,
				      .timestamp_seconds = 1.25,
				      .camera = { .fx_px = 100,
						  .fy_px = 120,
						  .cx_px = .25f,
						  .cy_px = .75f,
						  .sensor_width_mm = 36,
						  .sensor_height_mm = 24 } };
	struct strata_error err = { { 0 } };
	strata_context *ctx = NULL;
	float network;
	int ret = 1;

	if (argc != 3)
		return 2;
	options.model_path = argv[1];
	CHECK(strata_image_read(argv[2], &image, &err) == 0);
	CHECK(strata_open(&options, &ctx, &err) == 0);
	CHECK(!strcmp(strata_provider(ctx), "cpu"));
	CHECK(strata_run(ctx, &frame, 2, &result, &err) == -1);
	CHECK(strata_run(ctx, &frame, 1, &result, &err) == 0);
	CHECK(result.width == 2 && result.height == 2);
	CHECK(result.network_inverse_depth && result.inverse_metres &&
	      result.camera_z_metres);
	network = result.network_inverse_depth[0];
	CHECK(network > 0);
	CHECK(result.inverse_metres[0] == 2 * network / 100);
	CHECK(result.camera_z_metres[0] == 100 / (2 * network));
	CHECK(result.frame.index == 1001 &&
	      result.frame.timestamp_seconds == 1.25);
	CHECK(result.frame.camera.fx_px == 100 &&
	      result.frame.camera.fy_px == 120);
	CHECK(result.frame.camera.cx_px == .25f &&
	      result.frame.camera.cy_px == .75f);
	CHECK(result.frame.camera.sensor_width_mm == 36 &&
	      result.frame.camera.sensor_height_mm == 24);
	CHECK(result.frame.camera.focal_source == STRATA_FOCAL_PROVIDED);
	strata_result_free(&result);
	strata_unload(ctx);
	strata_unload(ctx);
	CHECK(strata_run(ctx, &frame, 1, &result, &err) == 0);
	CHECK(result.network_inverse_depth[0] == network);
	strata_close(ctx);
	ctx = NULL;
	CHECK(result.network_inverse_depth[0] == network);
	ret = 0;
	puts("session create/run/unload/reload/close, depth units and per-frame camera metadata passed");
out:
	if (ret)
		fprintf(stderr, "%s\n", err.message);
	strata_result_free(&result);
	strata_image_free(&image);
	strata_close(ctx);
	return ret;
}

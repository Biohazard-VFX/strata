#include "strata.h"
#include "expect.h"
#include <string.h>

int main(int argc, char **argv)
{
	float rgba[4 * 11 * 7];
	char companion[4096], moved[4096];
	struct strata_image image = { .width = 11, .height = 7, .rgba = rgba };
	struct strata_frame frame = {
		.image = &image,
		.colorspace = STRATA_COLOR_ACESCG,
		.camera = { .fx_px = 17,
			    .fy_px = 17,
			    .focal_source = STRATA_FOCAL_PROVIDED }
	};
	struct strata_options options = { .provider = "cpu" };
	struct strata_error err = { { 0 } };
	struct strata_result depth = { 0 }, plain = { 0 };
	struct strata_deep_result before = { 0 }, after = { 0 };
	strata_context *ctx = NULL;

	if (argc != 4)
		return 2;
	for (size_t i = 0; i < sizeof(rgba) / sizeof(*rgba); ++i)
		rgba[i] = (float)(i % 17) / 8;
	options.model_path = argv[1];
	if (!EXPECT(!strata_open(&options, &ctx, &err)))
		return 1;
	EXPECT(!strata_run(ctx, &frame, &plain, &err));
	EXPECT(strata_set_head(ctx, argv[3], &err) == -1);
	EXPECT(!strata_run(ctx, &frame, &depth, &err));
	EXPECT(depth.network_inverse_depth && plain.network_inverse_depth &&
	       !memcmp(depth.network_inverse_depth, plain.network_inverse_depth,
		       77 * sizeof(float)));
	EXPECT(strata_run_deep(ctx, &frame, &depth, &after, &err) == -1);
	EXPECT(strstr(err.message, "requires a head") != NULL);
	EXPECT(!strata_set_head(ctx, argv[2], &err));
	EXPECT(!strata_run_deep(ctx, &frame, &depth, &before, &err));
	for (int i = 0; i < 3; ++i) {
		const char *path = i == 0 ? argv[3] :
				   i == 1 ? "missing.onnx" :
					    "invalid";

		EXPECT(strata_set_head(ctx, path, &err) == -1);
		EXPECT(!strata_run_deep(ctx, &frame, &depth, &after, &err));
		EXPECT(before.sample_count == after.sample_count &&
		       before.sample_count == 308);
		if (before.sample_count == after.sample_count &&
		    before.sample_count) {
			EXPECT(!memcmp(before.rgba, after.rgba,
				       before.sample_count * 4 *
					       sizeof(float)));
			EXPECT(!memcmp(before.z, after.z,
				       before.sample_count * sizeof(float)));
		}
		EXPECT(!strata_run(ctx, &frame, &depth, &err));
		EXPECT(depth.network_inverse_depth &&
		       plain.network_inverse_depth &&
		       !memcmp(depth.network_inverse_depth,
			       plain.network_inverse_depth,
			       77 * sizeof(float)));
	}
	strata_unload(ctx);
	snprintf(companion, sizeof(companion), "%.*s.backbone.onnx",
		 (int)strlen(argv[2]) - 5, argv[2]);
	snprintf(moved, sizeof(moved), "%s.moved", argv[2]);
	if (EXPECT(!rename(companion, moved))) {
		EXPECT(!strata_run(ctx, &frame, &depth, &err));
		EXPECT(!rename(moved, companion));
	}
	EXPECT(!strata_run_deep(ctx, &frame, &depth, &after, &err));
	strata_result_free(&plain);
	strata_result_free(&depth);
	strata_deep_free(&before);
	strata_deep_free(&after);
	strata_close(ctx);
	return failures != 0;
}

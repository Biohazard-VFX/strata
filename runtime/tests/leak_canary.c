#include "strata.h"

static int leak(void)
{
	static float rgba[4], q[1] = { 1 };
	static const struct strata_image image = { .width = 1,
						   .height = 1,
						   .rgba = rgba };
	static const struct strata_result depth = {
		.width = 1,
		.height = 1,
		.frame.camera = { .fx_px = 1,
				  .fy_px = 1,
				  .focal_source = STRATA_FOCAL_PROVIDED },
		.network_inverse_depth = q
	};
	struct strata_deep_result deep = { 0 };
	struct strata_error err = { { 0 } };

	return strata_depth_to_deep(&image, &depth, &deep, &err);
}

int main(void)
{
	return leak();
}

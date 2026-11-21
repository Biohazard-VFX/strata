#include "strata.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                   \
	do {                                                               \
		if (!(condition)) {                                        \
			fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
				#condition);                               \
			ret = -1;                                          \
			goto out;                                          \
		}                                                          \
	} while (0)

static int layers_test(void)
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
	struct strata_layers layers = { 0 };
	struct strata_error err = { { 0 } };
	int expected[] = { -1, 1, 3, 5, 6, -1, -1, 2, -1, 4, -1, -1 };
	int ret = 0;

	for (size_t i = 0; i < 7; ++i) {
		float coverage = i == 1 ? .25f : 1;

		for (size_t c = 0; c < 3; ++c)
			rgba[i * 4 + c] = ((float)(i * 4 + c) - 10) * coverage;
		rgba[i * 4 + 3] = coverage;
	}
	CHECK(strata_deep_to_layers(&deep, 2, &layers, &err) == 0);
	CHECK(layers.width == 3 && layers.height == 2 &&
	      layers.layer_count == 2);
	for (size_t i = 0; i < 12; ++i) {
		if (expected[i] < 0) {
			CHECK(layers.z[i] == INFINITY &&
			      layers.present[i] == 0);
			for (size_t c = 0; c < 4; ++c)
				CHECK(layers.rgba[i * 4 + c] == 0);
		} else {
			size_t sample = (size_t)expected[i];

			CHECK(layers.z[i] == z[sample] &&
			      layers.present[i] == 1);
			CHECK(memcmp(layers.rgba + i * 4, rgba + sample * 4,
				     4 * sizeof(float)) == 0);
		}
	}
	CHECK(layers.frame.index == 42 && layers.frame.linear == 1);
	strata_layers_free(&layers);
	z[2] = 1;
	CHECK(strata_deep_to_layers(&deep, 3, &layers, &err) == 0);
	CHECK(layers.rgba[4] == rgba[4] && layers.rgba[28] == rgba[8]);
	strata_layers_free(&layers);
	z[2] = 2;
	counts[1] = 4;
	CHECK(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	counts[1] = 3;
	CHECK(strata_deep_to_layers(&deep, 0, &layers, &err) == -1);
	CHECK(strata_deep_to_layers(NULL, 1, &layers, &err) == -1);
	z[0] = NAN;
	CHECK(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	z[0] = 3;
	back[0] = 2;
	CHECK(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	memset(counts, 0, sizeof(counts));
	deep.sample_count = 0;
	deep.rgba = NULL;
	deep.z = NULL;
	deep.zback = NULL;
	CHECK(strata_deep_to_layers(&deep, 1, &layers, &err) == 0);
	for (size_t i = 0; i < 6; ++i)
		CHECK(layers.z[i] == INFINITY && layers.rgba[i * 4 + 3] == 0);
out:
	strata_layers_free(&layers);
	return ret;
}

static int conversion_test(void)
{
	float rgba[] = { -1, 2, 3, .2f, .1f, .2f, .3f, .4f };
	float z[] = { 2, INFINITY };
	struct strata_image image = { 2, 1, 1, rgba };
	struct strata_result depth = { .width = 2,
				       .height = 1,
				       .frame.camera.fx_px = 1000,
				       .camera_z_metres = z };
	struct strata_deep_result deep = { 0 };
	struct strata_error err = { { 0 } };
	int ret = 0;

	CHECK(strata_depth_to_deep(&image, &depth, &deep, &err) == 0);
	CHECK(deep.width == 2 && deep.height == 1 && deep.sample_count == 2);
	CHECK(deep.counts[0] == 1 && deep.counts[1] == 1);
	CHECK(deep.rgba[0] == -1 && deep.rgba[1] == 2 && deep.rgba[3] == 1);
	CHECK(deep.rgba[7] == 1 && deep.z[1] == INFINITY);
	CHECK(memcmp(deep.z, deep.zback, sizeof(z)) == 0);
	strata_deep_free(&deep);
	strata_deep_free(&deep);
	z[0] = NAN;
	CHECK(strata_depth_to_deep(&image, &depth, &deep, &err) == -1);
	CHECK(!deep.counts && !deep.rgba && !deep.z && !deep.zback);
	depth.width = 1;
	CHECK(strata_depth_to_deep(&image, &depth, &deep, &err) == -1);
out:
	strata_deep_free(&deep);
	return ret;
}

int main(void)
{
	return layers_test() || conversion_test();
}

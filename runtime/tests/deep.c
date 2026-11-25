#include "strata.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "expect.h"

static void layers_test(void)
{
	uint32_t counts[] = { 0, 3, 1, 2, 1, 0 };
	float rgba[28], z[] = { 3, 1, 2, 5, 6, 4, INFINITY };
	float back[] = { 3.5f, 1.5f, 2.5f, 5.5f, 6.5f, 4.5f, INFINITY };
	struct strata_deep_result deep = { .width = 3,
					   .height = 2,
					   .frame = { .index = 42,
						      .camera.fx_px = 1000 },
					   .sample_count = 7,
					   .counts = counts,
					   .rgba = rgba,
					   .z = z,
					   .zback = back };
	struct strata_layers layers = { 0 };
	struct strata_error err = { { 0 } };
	int expected[] = { -1, 1, 3, 5, 6, -1, -1, 2, -1, 4, -1, -1 };

	for (size_t i = 0; i < 7; ++i) {
		float coverage = i == 1 ? .25f : 1;

		for (size_t c = 0; c < 3; ++c)
			rgba[i * 4 + c] = ((float)(i * 4 + c) - 10) * coverage;
		rgba[i * 4 + 3] = coverage;
	}
	if (!EXPECT(strata_deep_to_layers(&deep, 2, &layers, &err) == 0))
		goto out;
	EXPECT(layers.width == 3 && layers.height == 2 &&
	       layers.layer_count == 2);
	for (size_t i = 0; i < 12; ++i) {
		if (expected[i] < 0) {
			EXPECT(layers.z[i] == INFINITY &&
			       layers.present[i] == 0);
			for (size_t c = 0; c < 4; ++c)
				EXPECT(layers.rgba[i * 4 + c] == 0);
		} else {
			size_t sample = (size_t)expected[i];

			EXPECT(layers.z[i] == z[sample] &&
			       layers.present[i] == 1);
			EXPECT(memcmp(layers.rgba + i * 4, rgba + sample * 4,
				      4 * sizeof(float)) == 0);
		}
	}
	EXPECT(layers.frame.index == 42);
	z[2] = 1;
	if (!EXPECT(strata_deep_to_layers(&deep, 3, &layers, &err) == 0))
		goto out;
	EXPECT(layers.rgba[4] == rgba[4] && layers.rgba[28] == rgba[8]);
	z[2] = 2;
	counts[1] = 4;
	EXPECT(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	EXPECT_ZERO(layers);
	counts[1] = 3;
	EXPECT(!strata_deep_to_layers(&deep, 1, &layers, &err));
	EXPECT(strata_deep_to_layers(&deep, 0, &layers, &err) == -1);
	EXPECT_ZERO(layers);
	EXPECT(!strata_deep_to_layers(&deep, 1, &layers, &err));
	EXPECT(strata_deep_to_layers(NULL, 1, &layers, &err) == -1);
	EXPECT_ZERO(layers);
	z[0] = NAN;
	EXPECT(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	EXPECT_ZERO(layers);
	z[0] = 3;
	back[0] = 2;
	EXPECT(strata_deep_to_layers(&deep, 1, &layers, &err) == -1);
	EXPECT_ZERO(layers);
	memset(counts, 0, sizeof(counts));
	deep.sample_count = 0;
	deep.rgba = NULL;
	deep.z = NULL;
	deep.zback = NULL;
	if (!EXPECT(strata_deep_to_layers(&deep, 1, &layers, &err) == 0))
		goto out;
	for (size_t i = 0; i < 6; ++i)
		EXPECT(layers.z[i] == INFINITY && layers.rgba[i * 4 + 3] == 0);
out:
	strata_layers_free(&layers);
}

static void conversion_test(void)
{
	float rgba[] = { -1, 2, 3, .2f, .1f, .2f, .3f, .4f };
	float inverse[] = { .5f, 1, .5f, .5f, .25f, .25f, 0, 0 };
	float alpha[] = { .25f, 1, 0, .5f, .5f, 0, 1, 1 };
	struct strata_image image = { .width = 2, .height = 1, .rgba = rgba };
	struct strata_frame_metadata frame = { .index = 17 };
	struct strata_deep_result deep = { 0 };
	struct strata_error err = { { 0 } };
	double sum[3] = { 0 }, remaining = 1;

	if (!EXPECT(!strata_prediction_to_deep(&image, &frame, 4, inverse,
					       alpha, &deep, &err)))
		goto out;
	EXPECT(deep.counts[0] == 3 && deep.counts[1] == 1 &&
	       deep.sample_count == 4);
	EXPECT(deep.z[0] == 2 && deep.z[1] == 4 && deep.z[2] == INFINITY);
	EXPECT(deep.z[3] == 1 && deep.frame.index == 17);
	for (size_t i = 0; i < 3; ++i) {
		for (size_t c = 0; c < 3; ++c)
			sum[c] += remaining * deep.rgba[i * 4 + c];
		remaining *= 1 - (double)deep.rgba[i * 4 + 3];
	}
	EXPECT(remaining == 0);
	for (size_t c = 0; c < 3; ++c)
		EXPECT(sum[c] == rgba[c]);
	inverse[4] = .5f;
	EXPECT(!strata_prediction_to_deep(&image, &frame, 4, inverse, alpha,
					  &deep, &err));
	EXPECT(deep.z[0] == deep.z[1] && deep.counts[0] == 3);
	inverse[4] = 1;
	EXPECT(strata_prediction_to_deep(&image, &frame, 4, inverse, alpha,
					 &deep, &err) == -1);
	EXPECT_ZERO(deep);
	inverse[4] = .25f;
	alpha[6] = .9f;
	EXPECT(strata_prediction_to_deep(&image, &frame, 4, inverse, alpha,
					 &deep, &err) == -1);
	EXPECT_ZERO(deep);
	alpha[6] = 1;
	inverse[0] = NAN;
	EXPECT(strata_prediction_to_deep(&image, &frame, 4, inverse, alpha,
					 &deep, &err) == -1);
	EXPECT_ZERO(deep);
out:
	strata_deep_free(&deep);
}

int main(void)
{
	layers_test();
	conversion_test();
	return failures != 0;
}

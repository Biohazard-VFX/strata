#include "../src/core/internal.h"
#include <inttypes.h>

#define SIDE 256

static size_t prepared_mismatches(const float *values, size_t count)
{
	static float rgba[SIDE * SIDE * 4], data[3 * SIDE * SIDE];
	const float mean[] = { STRATA_MEAN_R, STRATA_MEAN_G, STRATA_MEAN_B };
	const float std[] = { STRATA_STD_R, STRATA_STD_G, STRATA_STD_B };
	struct strata_image image = { SIDE, SIDE, 1, rgba, 0 };
	struct strata_tensor padded = { SIDE, SIDE, data };
	size_t wrong = 0;

	for (size_t i = 0; i < SIDE * SIDE * 4; ++i)
		rgba[i] = values[i % count];
	strata_prepare(&image, 0, &padded);
	for (size_t i = 0; i < SIDE * SIDE; ++i) {
		for (size_t c = 0; c < 3; ++c) {
			float expected =
				(strata_srgb(rgba[i * 4 + c]) - mean[c]) /
				std[c];

			wrong += memcmp(&expected, &data[c * SIDE * SIDE + i],
					sizeof(expected)) != 0;
		}
	}
	return wrong;
}

int main(void)
{
	static float values[1 << 16];
	uint32_t input, expected;
	size_t count = 0, wrong = 0, prepared;

	while (count < sizeof(values) / sizeof(values[0]) &&
	       scanf("%" SCNx32 " %" SCNx32, &input, &expected) == 2) {
		float encoded;
		uint32_t bits;

		memcpy(&values[count], &input, sizeof(input));
		encoded = strata_srgb(values[count]);
		memcpy(&bits, &encoded, sizeof(bits));
		wrong += bits != expected;
		++count;
	}
	prepared = count ? prepared_mismatches(values, count) : 0;
	printf("%zu sRGB samples, %zu differ; %zu prepared values differ\n",
	       count, wrong, prepared);
	return !count || wrong || prepared;
}

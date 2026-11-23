#include "../src/core/internal.h"
#include <inttypes.h>

int main(void)
{
	static float values[1 << 16];
	uint32_t input, expected;
	size_t count = 0, wrong = 0;

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
	printf("%zu sRGB samples, %zu differ\n", count, wrong);
	return !count || wrong;
}

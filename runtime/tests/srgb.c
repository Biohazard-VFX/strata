#include "../src/core/internal.h"
#include <inttypes.h>

int main(void)
{
	uint32_t input, expected;
	size_t count = 0, wrong = 0;

	while (scanf("%" SCNx32 " %" SCNx32, &input, &expected) == 2) {
		float value, encoded;
		uint32_t bits;

		memcpy(&value, &input, sizeof(value));
		encoded = strata_srgb(value);
		memcpy(&bits, &encoded, sizeof(bits));
		wrong += bits != expected;
		++count;
	}
	printf("%zu sRGB samples, %zu differ\n", count, wrong);
	return !count || wrong;
}

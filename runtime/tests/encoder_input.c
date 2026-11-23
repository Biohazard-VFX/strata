#include "../src/core/internal.h"

int main(int argc, char **argv)
{
	float rgba[32 * 4] = { 0 }, data[3 * 32 * 32];
	struct strata_image image = { 32, 1, 1, rgba, 0 };
	struct strata_tensor padded = { 32, 32, data };
	float exposure;

	if (argc != 2 || sscanf(argv[1], "%f", &exposure) != 1)
		return 1;
	for (size_t i = 0; i < 32; ++i) {
		if (scanf("%f", &rgba[i * 4]) != 1)
			return 1;
		for (size_t c = 1; c < 3; ++c)
			rgba[i * 4 + c] = rgba[i * 4];
	}
	strata_prepare(&image, exposure, &padded);
	return fwrite(data, sizeof(float), 3 * 32 * 32, stdout) != 3 * 32 * 32;
}

#include "../src/core/internal.h"

int main(int argc, char **argv)
{
	float rgba[1023 * 4] = { 0 }, data[3 * 32 * 1024];
	struct strata_image image = { .width = 1023,
				      .height = 1,
				      .rgba = rgba };
	struct strata_tensor padded = { 1024, 32, data };
	double exposure;

	if (argc != 2 || sscanf(argv[1], "%lf", &exposure) != 1)
		return 1;
	for (size_t i = 0; i < 1023; ++i) {
		if (scanf("%f", &rgba[i * 4]) != 1)
			return 1;
		for (size_t c = 1; c < 3; ++c)
			rgba[i * 4 + c] = rgba[i * 4];
	}
	strata_prepare(&image, exposure, &padded);
	return fwrite(data, sizeof(float), 3 * 32 * 1024, stdout) !=
	       3 * 32 * 1024;
}

#include "../src/core/internal.h"

int main(int argc, char **argv)
{
	float rgba[16] = {
		1, 0, 0, 1, 0, 1, 0, .5f, 0, 0, 1, .25f, -1, 2, 4, 1
	};
	struct strata_image input = { .width = 4, .height = 1, .rgba = rgba };
	struct strata_image model = { 0 };
	struct strata_frame frame = { .image = &input };
	struct strata_error err = { { 0 } };
	int ret;

	if (argc != 2)
		return 2;
	frame.colorspace = strata_colorspace_parse(argv[1]);
	if (strata_model_image(&frame, &model, &err)) {
		fprintf(stderr, "%s\n", err.message);
		return 1;
	}
	ret = fwrite(model.rgba, sizeof(float), 16, stdout) != 16;
	strata_image_free(&model);
	return ret;
}

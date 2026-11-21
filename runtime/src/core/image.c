#include "internal.h"
#include <stdarg.h>
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

int strata_fail(struct strata_error *err, const char *fmt, ...)
{
	va_list ap;

	if (err) {
		va_start(ap, fmt);
		vsnprintf(err->message, sizeof(err->message), fmt, ap);
		va_end(ap);
	}
	return -1;
}

int strata_dimensions(int w, int h, struct strata_error *err)
{
	if (w <= 0 || h <= 0 || w > 32768 || h > 32768 ||
	    (size_t)w * (size_t)h > SIZE_MAX / (16 * sizeof(float)))
		return strata_fail(err, "invalid image dimensions %d x %d", w,
				   h);
	return 0;
}

float strata_srgb(float x)
{
	x = fmaxf(0, fminf(1, x));
	return x <= STRATA_SRGB_THRESHOLD ?
		       STRATA_SRGB_LINEAR_SCALE * x :
		       STRATA_SRGB_POWER_SCALE *
				       powf(x, 1 / STRATA_SRGB_GAMMA) -
			       STRATA_SRGB_POWER_OFFSET;
}

static int read_bytes(const char *path, struct strata_image *image,
		      struct strata_error *err)
{
	int w, h, channels;
	unsigned char *bytes = stbi_load(path, &w, &h, &channels, 3);
	size_t pixels;
	int ret = -1;

	if (!bytes)
		return strata_fail(err, "image decode: %s",
				   stbi_failure_reason());
	if (strata_dimensions(w, h, err))
		goto out;
	pixels = (size_t)w * (size_t)h;
	image->rgba = malloc(pixels * 4 * sizeof(float));
	if (!image->rgba) {
		strata_fail(err, "out of memory");
		goto out;
	}
	image->width = w;
	image->height = h;
	for (size_t i = 0; i < pixels; ++i) {
		for (size_t c = 0; c < 3; ++c)
			image->rgba[i * 4 + c] = bytes[i * 3 + c] / 255.0f;
		image->rgba[i * 4 + 3] = 1;
	}
	ret = 0;
out:
	stbi_image_free(bytes);
	return ret;
}

int strata_image_read(const char *path, struct strata_image *image,
		      struct strata_error *err)
{
	unsigned char magic[4];
	FILE *file;
	size_t count;

	if (!path || !image)
		return strata_fail(err, "null image path/output");
	memset(image, 0, sizeof(*image));
	file = fopen(path, "rb");
	if (!file)
		return strata_fail(err, "cannot open image: %s", path);
	count = fread(magic, 1, 4, file);
	fclose(file);
	if (count == 4 && magic[0] == 0x76 && magic[1] == 0x2f &&
	    magic[2] == 0x31 && magic[3] == 1)
		return strata_read_exr(path, image, err);
	return read_bytes(path, image, err);
}

void strata_image_free(struct strata_image *image)
{
	if (!image)
		return;
	free(image->rgba);
	memset(image, 0, sizeof(*image));
}

void strata_result_free(struct strata_result *result)
{
	if (!result)
		return;
	free(result->network_inverse_depth);
	free(result->inverse_metres);
	free(result->camera_z_metres);
	memset(result, 0, sizeof(*result));
}

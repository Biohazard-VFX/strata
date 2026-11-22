#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

double strata_milliseconds(void)
{
	struct timespec time;

	clock_gettime(CLOCK_MONOTONIC, &time);
	return 1000.0 * (double)time.tv_sec + (double)time.tv_nsec / 1000000.0;
}

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
	if (w <= 0 || h <= 0 || w > 32768 || h > 32768)
		return strata_fail(err, "invalid image dimensions %d x %d", w,
				   h);
	return 0;
}

char *strata_sibling_path(const char *model, const char *suffix)
{
	size_t length = strlen(model), extra = strlen(suffix);
	size_t stem = length >= 5 && !strcmp(model + length - 5, ".onnx") ?
			      length - 5 :
			      length;
	char *path = malloc(stem + extra + 1);

	if (!path)
		return NULL;
	memcpy(path, model, stem);
	memcpy(path + stem, suffix, extra + 1);
	return path;
}

int strata_atomic_target(const char *path, char **temporary,
			 struct strata_error *err)
{
	static atomic_uint serial;
	size_t length = strlen(path) + 48;
	char *name = malloc(length);
	int fd = -1;

	if (!name)
		return strata_fail(err, "out of memory");
	for (int attempt = 0; fd < 0 && attempt < 16; ++attempt) {
		snprintf(name, length, "%s.%ld.%u.tmp", path, (long)getpid(),
			 atomic_fetch_add(&serial, 1u));
		fd = open(name, O_WRONLY | O_CREAT | O_EXCL, 0666);
		if (fd < 0 && errno != EEXIST)
			break;
	}
	if (fd < 0) {
		strata_fail(err, "cannot create temporary output for %s", path);
		free(name);
		return -1;
	}
	close(fd);
	*temporary = name;
	return 0;
}

int strata_atomic_finish(const char *path, char *temporary, int ret,
			 struct strata_error *err)
{
	if (!ret && rename(temporary, path))
		ret = strata_fail(err, "cannot replace %s", path);
	if (ret)
		unlink(temporary);
	free(temporary);
	return ret;
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
	image->owned = 1;
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

	if (!image)
		return strata_fail(err, "missing image output");
	strata_image_free(image);
	if (!path)
		return strata_fail(err, "missing image path");
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
	if (image->owned)
		free(image->rgba);
	memset(image, 0, sizeof(*image));
}

void strata_result_free(struct strata_result *result)
{
	if (!result)
		return;
	free(result->network_inverse_depth);
	memset(result, 0, sizeof(*result));
}

#include "internal.h"

void strata_deep_free(struct strata_deep_result *deep)
{
	if (!deep)
		return;
	free(deep->counts);
	free(deep->rgba);
	free(deep->z);
	free(deep->zback);
	memset(deep, 0, sizeof(*deep));
}

void strata_layers_free(struct strata_layers *layers)
{
	if (!layers)
		return;
	free(layers->present);
	free(layers->rgba);
	free(layers->z);
	memset(layers, 0, sizeof(*layers));
}

int strata_deep_validate(const struct strata_deep_result *deep,
			 struct strata_error *err)
{
	size_t offset = 0, pixels;

	if (!deep || !deep->counts ||
	    (deep->sample_count && (!deep->rgba || !deep->z || !deep->zback)))
		return strata_fail(err, "missing deep buffers");
	if (strata_dimensions(deep->width, deep->height, err))
		return -1;
	if (deep->sample_count > SIZE_MAX / (4 * sizeof(float)))
		return strata_fail(err, "deep sample allocation overflow");
	pixels = (size_t)deep->width * (size_t)deep->height;
	for (size_t i = 0; i < pixels; ++i) {
		if (deep->counts[i] > deep->sample_count - offset)
			return strata_fail(err,
					   "deep counts exceed sample buffers");
		offset += deep->counts[i];
	}
	if (offset != deep->sample_count)
		return strata_fail(err,
				   "deep counts do not match sample count");
	for (size_t i = 0; i < offset; ++i) {
		float coverage = deep->rgba[i * 4 + 3];

		if (!isfinite(coverage) || coverage < 0 || coverage > 1)
			return strata_fail(
				err, "deep alpha must be coverage in [0,1]");
		if (isnan(deep->z[i]) || deep->z[i] < 0 ||
		    isnan(deep->zback[i]) || deep->zback[i] < deep->z[i])
			return strata_fail(err,
					   "invalid deep Z/ZBack interval");
	}
	return 0;
}

int strata_depth_to_deep(const struct strata_image *image,
			 const struct strata_result *depth,
			 struct strata_deep_result *deep,
			 struct strata_error *err)
{
	size_t pixels;
	int ret = -1;

	if (!deep)
		return strata_fail(err, "missing deep output");
	memset(deep, 0, sizeof(*deep));
	if (!image || !image->rgba || !depth || !depth->network_inverse_depth ||
	    image->width != depth->width || image->height != depth->height)
		return strata_fail(err, "colour/depth dimensions differ");
	if (strata_dimensions(image->width, image->height, err) ||
	    strata_camera_validate(&depth->frame.camera, 1, err))
		return -1;
	pixels = (size_t)image->width * (size_t)image->height;
	deep->width = image->width;
	deep->height = image->height;
	deep->frame = depth->frame;
	deep->frame.linear = image->linear;
	deep->sample_count = pixels;
	deep->counts = malloc(pixels * sizeof(uint32_t));
	deep->rgba = malloc(pixels * 4 * sizeof(float));
	deep->z = malloc(pixels * sizeof(float));
	deep->zback = malloc(pixels * sizeof(float));
	if (!deep->counts || !deep->rgba || !deep->z || !deep->zback) {
		strata_fail(err, "out of memory");
		goto out;
	}
	for (size_t i = 0; i < pixels; ++i) {
		deep->counts[i] = 1;
		memcpy(deep->rgba + i * 4, image->rgba + i * 4,
		       3 * sizeof(float));
		deep->rgba[i * 4 + 3] = 1;
		deep->z[i] = strata_camera_z_metres(depth, i);
		deep->zback[i] = deep->z[i];
	}
	ret = strata_deep_validate(deep, err);
out:
	if (ret)
		strata_deep_free(deep);
	return ret;
}

static void insert_sample(struct strata_layers *layers, size_t pixel,
			  size_t seen, float z, const float *rgba)
{
	size_t pixels = (size_t)layers->width * (size_t)layers->height;
	size_t count = (size_t)layers->layer_count;
	size_t used = seen < count ? seen : count;
	size_t position = 0;

	while (position < used && z >= layers->z[position * pixels + pixel])
		++position;
	if (position == count)
		return;
	for (size_t i = used < count ? used : count - 1; i > position; --i) {
		size_t target = i * pixels + pixel;
		size_t source = (i - 1) * pixels + pixel;

		layers->z[target] = layers->z[source];
		layers->present[target] = layers->present[source];
		memcpy(layers->rgba + target * 4, layers->rgba + source * 4,
		       4 * sizeof(float));
	}
	layers->z[position * pixels + pixel] = z;
	layers->present[position * pixels + pixel] = 1;
	memcpy(layers->rgba + (position * pixels + pixel) * 4, rgba,
	       4 * sizeof(float));
}

int strata_deep_to_layers(const struct strata_deep_result *deep, int count,
			  struct strata_layers *layers,
			  struct strata_error *err)
{
	size_t pixels, samples, offset = 0;
	int ret = -1;

	if (!layers)
		return strata_fail(err, "missing layer output");
	memset(layers, 0, sizeof(*layers));
	if (count < 1)
		return strata_fail(err, "layer count must be positive");
	if (strata_deep_validate(deep, err))
		return -1;
	pixels = (size_t)deep->width * (size_t)deep->height;
	if (pixels > SIZE_MAX / (size_t)count ||
	    pixels * (size_t)count > SIZE_MAX / (4 * sizeof(float)))
		return strata_fail(err, "flat layer allocation overflow");
	samples = pixels * (size_t)count;
	layers->width = deep->width;
	layers->height = deep->height;
	layers->frame = deep->frame;
	layers->layer_count = count;
	layers->present = calloc(samples, sizeof(uint8_t));
	layers->rgba = calloc(samples * 4, sizeof(float));
	layers->z = malloc(samples * sizeof(float));
	if (!layers->present || !layers->rgba || !layers->z) {
		strata_fail(err, "out of memory");
		goto out;
	}
	for (size_t i = 0; i < samples; ++i)
		layers->z[i] = INFINITY;
	for (size_t i = 0; i < pixels; ++i) {
		for (size_t j = 0; j < deep->counts[i]; ++j)
			insert_sample(layers, i, j, deep->z[offset + j],
				      deep->rgba + (offset + j) * 4);
		offset += deep->counts[i];
	}
	ret = 0;
out:
	if (ret)
		strata_layers_free(layers);
	return ret;
}

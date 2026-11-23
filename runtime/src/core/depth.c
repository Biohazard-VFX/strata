#include "internal.h"
#include <float.h>

static int tensor_alloc(struct strata_tensor *tensor, int w, int h,
			struct strata_error *err)
{
	if (strata_dimensions(w, h, err))
		return -1;
	tensor->data =
		malloc((size_t)3 * (size_t)w * (size_t)h * sizeof(float));
	if (!tensor->data)
		return strata_fail(err, "out of memory");
	tensor->width = w;
	tensor->height = h;
	return 0;
}

static void tensor_free(struct strata_tensor *tensor)
{
	free(tensor->data);
	memset(tensor, 0, sizeof(*tensor));
}

static float srgb(double v)
{
	if (v <= STRATA_SRGB_THRESHOLD)
		return (float)(STRATA_SRGB_LINEAR_SCALE * v);
	return (float)(STRATA_SRGB_POWER_SCALE * pow(v, 1 / STRATA_SRGB_GAMMA) -
		       STRATA_SRGB_POWER_OFFSET);
}

float strata_srgb(float x)
{
	return srgb(x);
}

int strata_padded_size(int w, int h, int *pw, int *ph, struct strata_error *err)
{
	if (strata_dimensions(w, h, err))
		return -1;
	*pw = (w + STRATA_INPUT_MULTIPLE - 1) / STRATA_INPUT_MULTIPLE *
	      STRATA_INPUT_MULTIPLE;
	*ph = (h + STRATA_INPUT_MULTIPLE - 1) / STRATA_INPUT_MULTIPLE *
	      STRATA_INPUT_MULTIPLE;
	if (*pw > STRATA_MAX_WIDTH || *ph > STRATA_MAX_HEIGHT)
		return strata_fail(
			err, "input exceeds the model contract dimensions");
	return 0;
}

static float clamp_unit(float value)
{
	value = value > 0 ? value : 0;
	return value < 1 ? value : 1;
}

static float shoulder(double x)
{
	if (!(x > 0))
		return 0;
	if (!isfinite(x))
		return 1;
	if (x > 0.6)
		x = 0.6 + 0.4 * (x - 0.6) / (0.4 + x - 0.6);
	return srgb(x);
}

static size_t nonfinite_pixels(const float *rgba, size_t w)
{
	size_t count = 0;

	for (size_t x = 0; x < w; ++x) {
		const float *p = rgba + x * 4;

		count += (size_t)!((fabsf(p[0]) <= FLT_MAX) &
				   (fabsf(p[1]) <= FLT_MAX) &
				   (fabsf(p[2]) <= FLT_MAX));
	}
	return count;
}

static void prepare_row(const float *rgba, size_t w, double gain, int linear,
			float *const out[3])
{
	const float mean[] = { STRATA_MEAN_R, STRATA_MEAN_G, STRATA_MEAN_B };
	const float std[] = { STRATA_STD_R, STRATA_STD_G, STRATA_STD_B };

	if (!linear) {
		for (size_t c = 0; c < 3; ++c) {
			for (size_t x = 0; x < w; ++x)
				out[c][x] = (clamp_unit(rgba[x * 4 + c]) -
					     mean[c]) /
					    std[c];
		}
		return;
	}
	for (size_t c = 0; c < 3; ++c) {
		for (size_t x = 0; x < w; ++x) {
			double value = rgba[x * 4 + c] * gain;

			out[c][x] = (shoulder(value) - mean[c]) / std[c];
		}
	}
}

size_t strata_prepare(const struct strata_image *image, double exposure,
		      const struct strata_tensor *padded)
{
	size_t w = (size_t)image->width, h = (size_t)image->height;
	size_t pw = (size_t)padded->width, ph = (size_t)padded->height;
	size_t plane = pw * ph, nonfinite = 0;
	double gain = 0.72 * pow(2.0, (double)exposure);
	for (size_t y = 0; y < h; ++y) {
		const float *row = image->rgba + y * w * 4;
		float *out[3];

		for (size_t c = 0; c < 3; ++c)
			out[c] = padded->data + c * plane + y * pw;
		nonfinite += nonfinite_pixels(row, w);
		prepare_row(row, w, gain, image->linear, out);
		for (size_t c = 0; c < 3; ++c) {
			for (size_t x = w; x < pw; ++x)
				out[c][x] = out[c][w - 1];
		}
	}
	for (size_t c = 0; c < 3; ++c) {
		float *last = padded->data + c * plane + (h - 1) * pw;

		for (size_t y = h; y < ph; ++y)
			memcpy(last + (y - h + 1) * pw, last,
			       pw * sizeof(float));
	}
	return nonfinite;
}

static float filter_weight(int sample, float center, float support)
{
	return fmaxf(0, 1 - fabsf(((float)sample + .5f - center) / support));
}

static float filtered(const float *in, size_t stride, int first, int end,
		      float center, float support)
{
	float total = 0, value = 0;

	for (int i = first; i < end; ++i)
		total += filter_weight(i, center, support);
	for (int i = first; i < end; ++i)
		value += in[(size_t)i * stride] *
			 (filter_weight(i, center, support) / total);
	return value;
}

static void resize_axis(const struct strata_tensor *in,
			struct strata_tensor *out, int axis)
{
	int src = axis ? in->height : in->width;
	int dst = axis ? out->height : out->width;
	float scale = (float)src / (float)dst;
	float support = fmaxf(1, scale);
	size_t h = (size_t)in->height, w = (size_t)in->width;
	size_t oh = (size_t)out->height, ow = (size_t)out->width;

	for (size_t c = 0; c < 3; ++c) {
		for (size_t y = 0; y < oh; ++y) {
			for (size_t x = 0; x < ow; ++x) {
				float center =
					((float)(axis ? y : x) + .5f) * scale;
				int first = (int)(center - support + .5f);
				int end = (int)(center + support + .5f);
				size_t base = c * h * w + (axis ? x : y * w);

				if (first < 0)
					first = 0;
				if (end > src)
					end = src;
				out->data[(c * oh + y) * ow + x] =
					filtered(in->data + base, axis ? w : 1,
						 first, end, center, support);
			}
		}
	}
}

static void fov_size(int w, int h, int *rw, int *rh, int *cw, int *ch)
{
	int lo = h < w ? h : w;
	int short_side = lo / 32 * 32;
	double scale;

	if (short_side < 32)
		short_side = 32;
	if (short_side > STRATA_FOV_SHORT_MAX)
		short_side = STRATA_FOV_SHORT_MAX;
	scale = (double)short_side / lo;
	*rh = (int)nearbyint(h * scale / 32) * 32;
	*rw = (int)nearbyint(w * scale / 32) * 32;
	if (*rh < 32)
		*rh = 32;
	if (*rw < 32)
		*rw = 32;
	*ch = *rh;
	*cw = *rw;
	lo = *rh < *rw ? *rh : *rw;
	if ((*rh > *rw ? *rh : *rw) > (double)STRATA_FOV_ASPECT_MAX * lo) {
		int keep = (int)((double)STRATA_FOV_ASPECT_MAX * lo) / 32 * 32;

		if (*rw > *rh)
			*cw = keep;
		else
			*ch = keep;
	}
}

static void crop_input(const struct strata_tensor *padded,
		       struct strata_tensor *input)
{
	size_t w = (size_t)input->width, h = (size_t)input->height;
	size_t pw = (size_t)padded->width, ph = (size_t)padded->height;

	for (size_t c = 0; c < 3; ++c) {
		for (size_t y = 0; y < h; ++y)
			memcpy(input->data + (c * h + y) * w,
			       padded->data + (c * ph + y) * pw,
			       w * sizeof(float));
	}
}

static void center_crop(const struct strata_tensor *resized,
			struct strata_tensor *crop)
{
	size_t rw = (size_t)resized->width, rh = (size_t)resized->height;
	size_t cw = (size_t)crop->width, ch = (size_t)crop->height;

	for (size_t c = 0; c < 3; ++c) {
		for (size_t y = 0; y < ch; ++y)
			memcpy(crop->data + (c * ch + y) * cw,
			       resized->data +
				       (c * rh + y + (rh - ch) / 2) * rw +
				       (rw - cw) / 2,
			       cw * sizeof(float));
	}
}

int strata_fov_input(const struct strata_tensor *padded, int w, int h,
		     struct strata_tensor *crop, struct strata_error *err)
{
	struct strata_tensor input = { 0 }, horizontal = { 0 }, resized = { 0 };
	int rh, rw, ch, cw, ret = -1;

	fov_size(w, h, &rw, &rh, &cw, &ch);
	if (tensor_alloc(&input, w, h, err) ||
	    tensor_alloc(&horizontal, rw, h, err) ||
	    tensor_alloc(&resized, rw, rh, err) ||
	    tensor_alloc(crop, cw, ch, err))
		goto out;
	crop_input(padded, &input);
	resize_axis(&input, &horizontal, 0);
	resize_axis(&horizontal, &resized, 1);
	center_crop(&resized, crop);
	ret = 0;
out:
	tensor_free(&input);
	tensor_free(&horizontal);
	tensor_free(&resized);
	if (ret)
		tensor_free(crop);
	return ret;
}

int strata_focal(const float *angles, int w, int h, float *focal,
		 struct strata_error *err)
{
	for (size_t i = 0; i < 2; ++i) {
		if (!(angles[i] > 0 && angles[i] < 180))
			return strata_fail(
				err,
				"invalid estimated field of view: %g degrees",
				(double)angles[i]);
	}
	*focal = .5f * (float)(w < h ? w : h) /
		 tanf(angles[w < h ? 0 : 1] * .008726646259971648f);
	return 0;
}

int strata_postprocess(const float *prediction, int stride,
		       struct strata_result *result, struct strata_error *err)
{
	size_t w = (size_t)result->width, h = (size_t)result->height;
	float *out = malloc(w * h * sizeof(float));

	if (!out)
		return strata_fail(err, "out of memory");
	for (size_t y = 0; y < h; ++y) {
		const float *row = prediction + y * (size_t)stride;

		for (size_t x = 0; x < w; ++x) {
			if (!isfinite(row[x]) || row[x] < 0) {
				free(out);
				return strata_fail(
					err, "invalid network inverse depth");
			}
		}
		memcpy(out + y * w, row, w * sizeof(float));
	}
	result->network_inverse_depth = out;
	return 0;
}

float strata_camera_z_metres(const struct strata_result *result, size_t pixel)
{
	return result->frame.camera.fx_px /
	       ((float)result->width * result->network_inverse_depth[pixel]);
}

float strata_inverse_metres(const struct strata_result *result, size_t pixel)
{
	return (float)result->width * result->network_inverse_depth[pixel] /
	       result->frame.camera.fx_px;
}

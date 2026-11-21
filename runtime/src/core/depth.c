#include "internal.h"

int strata_tensor_alloc(struct strata_tensor *tensor, int w, int h,
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

void strata_tensor_free(struct strata_tensor *tensor)
{
	free(tensor->data);
	memset(tensor, 0, sizeof(*tensor));
}

int strata_normalize(const struct strata_image *image, float exposure,
		     struct strata_tensor *tensor, struct strata_error *err)
{
	const float mean[] = { STRATA_MEAN_R, STRATA_MEAN_G, STRATA_MEAN_B };
	const float std[] = { STRATA_STD_R, STRATA_STD_G, STRATA_STD_B };
	float gain = exp2f(exposure);
	int ret = -1;
	size_t pixels;

	if (strata_tensor_alloc(tensor, image->width, image->height, err))
		return -1;
	pixels = (size_t)image->width * (size_t)image->height;
	for (size_t c = 0; c < 3; ++c) {
		for (size_t i = 0; i < pixels; ++i) {
			float value = image->rgba[i * 4 + c];

			if (!isfinite(value)) {
				strata_fail(err, "non-finite input pixel");
				goto out;
			}
			if (image->linear)
				value = strata_srgb(value * gain);
			tensor->data[c * pixels + i] =
				(value - mean[c]) / std[c];
		}
	}
	ret = 0;
out:
	if (ret)
		strata_tensor_free(tensor);
	return ret;
}

int strata_pad(const struct strata_tensor *input, struct strata_tensor *padded,
	       struct strata_error *err)
{
	int ph = (input->height + STRATA_INPUT_MULTIPLE - 1) /
		 STRATA_INPUT_MULTIPLE * STRATA_INPUT_MULTIPLE;
	int pw = (input->width + STRATA_INPUT_MULTIPLE - 1) /
		 STRATA_INPUT_MULTIPLE * STRATA_INPUT_MULTIPLE;
	size_t h = (size_t)input->height, w = (size_t)input->width;

	if (pw > STRATA_MAX_WIDTH || ph > STRATA_MAX_HEIGHT)
		return strata_fail(
			err, "input exceeds the model contract dimensions");
	if (strata_tensor_alloc(padded, pw, ph, err))
		return -1;
	for (size_t c = 0; c < 3; ++c) {
		for (size_t y = 0; y < (size_t)ph; ++y) {
			for (size_t x = 0; x < (size_t)pw; ++x) {
				size_t sy = y < h ? y : h - 1;
				size_t sx = x < w ? x : w - 1;

				padded->data[(c * (size_t)ph + y) * (size_t)pw +
					     x] =
					input->data[(c * h + sy) * w + sx];
			}
		}
	}
	return 0;
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

int strata_fov_input(const struct strata_tensor *input,
		     struct strata_tensor *crop, struct strata_error *err)
{
	struct strata_tensor horizontal = { 0 }, resized = { 0 };
	int rh, rw, ch, cw, ret = -1;

	fov_size(input->width, input->height, &rw, &rh, &cw, &ch);
	if (strata_tensor_alloc(&horizontal, rw, input->height, err) ||
	    strata_tensor_alloc(&resized, rw, rh, err) ||
	    strata_tensor_alloc(crop, cw, ch, err))
		goto out;
	resize_axis(input, &horizontal, 0);
	resize_axis(&horizontal, &resized, 1);
	for (size_t c = 0; c < 3; ++c) {
		for (size_t y = 0; y < (size_t)ch; ++y) {
			size_t src =
				(c * (size_t)rh + y + (size_t)(rh - ch) / 2) *
					(size_t)rw +
				(size_t)(rw - cw) / 2;
			size_t dst = (c * (size_t)ch + y) * (size_t)cw;

			memcpy(crop->data + dst, resized.data + src,
			       (size_t)cw * sizeof(float));
		}
	}
	ret = 0;
out:
	strata_tensor_free(&horizontal);
	strata_tensor_free(&resized);
	if (ret)
		strata_tensor_free(crop);
	return ret;
}

float strata_focal(const float *angles, int w, int h)
{
	return .5f * (float)(w < h ? w : h) /
	       tanf(angles[w < h ? 0 : 1] * .008726646259971648f);
}

int strata_postprocess(const float *prediction, int stride, int w, int h,
		       float focal, struct strata_result *result,
		       struct strata_error *err)
{
	size_t pixels = (size_t)w * (size_t)h;
	int ret = -1;

	if (!isfinite(focal) || focal <= 0)
		return strata_fail(err, "invalid focal length");
	result->network_inverse_depth = malloc(pixels * sizeof(float));
	result->camera_z_metres = malloc(pixels * sizeof(float));
	result->inverse_metres = malloc(pixels * sizeof(float));
	if (!result->network_inverse_depth || !result->camera_z_metres ||
	    !result->inverse_metres) {
		strata_fail(err, "out of memory");
		goto out;
	}
	result->width = w;
	result->height = h;
	result->frame.camera.fx_px = focal;
	result->frame.camera.fy_px = focal;
	result->frame.camera.cx_px = (float)w / 2;
	result->frame.camera.cy_px = (float)h / 2;
	for (size_t y = 0; y < (size_t)h; ++y) {
		for (size_t x = 0; x < (size_t)w; ++x) {
			size_t i = y * (size_t)w + x;
			float inverse = prediction[y * (size_t)stride + x];

			if (!isfinite(inverse) || inverse < 0) {
				strata_fail(err,
					    "invalid network inverse depth");
				goto out;
			}
			result->network_inverse_depth[i] = inverse;
			result->camera_z_metres[i] =
				focal / ((float)w * inverse);
			result->inverse_metres[i] = (float)w * inverse / focal;
		}
	}
	ret = 0;
out:
	if (ret)
		strata_result_free(result);
	return ret;
}

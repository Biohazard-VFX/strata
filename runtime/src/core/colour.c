#include "internal.h"

static const struct {
	const char *name;
	double matrix[9];
	float chromaticities[8];
} spaces[] = {
	{ NULL, { 0 }, { 0 } },
	{ "acescg",
	  { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
	  { .713f, .293f, .165f, .830f, .128f, .044f, .32168f, .33767f } },
	{ "aces2065-1",
	  { 1.451439316145665, -.2365107468937402, -.2149285692519253,
	    -.07655377339602057, 1.176229699833573, -.09967592643755223,
	    .008316148425697719, -.006032449791021028, .9977163013653234 },
	  { .7347f, .2653f, 0, 1, .0001f, -.077f, .32168f, .33767f } },
	{ "lin-rec709",
	  { .6130974024011883, .33952314618410545, .047379451414707265,
	    .0701937224695816, .9163538790573413, .013452398473073869,
	    .020615592882227002, .1095697729381357, .8698146341796399 },
	  { .64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f } },
	{ "lin-rec2020",
	  { .9748949779244203, .019599108637005452, .005505913438576727,
	    .0021795627977038495, .9955354688932179, .0022849683090743005,
	    .004797239683772719, .024532016634589246, .9706707436816335 },
	  { .708f, .292f, .170f, .797f, .131f, .046f, .3127f, .3290f } }
};

const char *strata_colorspace_name(enum strata_colorspace space)
{
	return space >= STRATA_COLOR_ACESCG &&
			       space <= STRATA_COLOR_LIN_REC2020 ?
		       spaces[space].name :
		       NULL;
}

enum strata_colorspace strata_colorspace_parse(const char *name)
{
	for (int i = STRATA_COLOR_ACESCG; name && i <= STRATA_COLOR_LIN_REC2020;
	     ++i) {
		if (!strcmp(name, spaces[i].name))
			return (enum strata_colorspace)i;
	}
	return STRATA_COLOR_UNSET;
}

const float *strata_chromaticities(enum strata_colorspace space)
{
	return strata_colorspace_name(space) ? spaces[space].chromaticities :
					       NULL;
}

int strata_model_image(const struct strata_frame *frame,
		       struct strata_image *image, struct strata_error *err)
{
	const struct strata_image *source;
	size_t pixels;

	if (!frame || !frame->image || !frame->image->rgba)
		return strata_fail(err, "invalid frame");
	if (!strata_colorspace_name(frame->colorspace))
		return strata_fail(
			err,
			"input colorspace required: acescg, aces2065-1, lin-rec709, lin-rec2020");
	source = frame->image;
	if (strata_dimensions(source->width, source->height, err))
		return -1;
	pixels = (size_t)source->width * (size_t)source->height;
	*image = *source;
	image->owned = 0;
	if (frame->colorspace == STRATA_COLOR_ACESCG)
		return 0;
	image->rgba = malloc(pixels * 4 * sizeof(float));
	if (!image->rgba)
		return strata_fail(err, "out of memory");
	image->owned = 1;
	for (size_t i = 0; i < pixels; ++i) {
		const float *in = source->rgba + i * 4;
		float *out = image->rgba + i * 4;
		const double *m = spaces[frame->colorspace].matrix;

		for (size_t c = 0; c < 3; ++c)
			out[c] = (float)(m[c * 3] * in[0] +
					 m[c * 3 + 1] * in[1] +
					 m[c * 3 + 2] * in[2]);
		out[3] = in[3];
	}
	return 0;
}

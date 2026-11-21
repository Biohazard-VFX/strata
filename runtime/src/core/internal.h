#ifndef STRATA_INTERNAL_H
#define STRATA_INTERNAL_H
#include "strata.h"
#include "strata_contract.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

int strata_camera_validate(const struct strata_camera *camera,
			   struct strata_error *err);
int strata_pose_validate(const struct strata_pose *pose,
			 struct strata_error *err);
int strata_deep_validate(const struct strata_deep_result *deep,
			 struct strata_error *err);
int strata_check_manifest(const char *model, struct strata_error *err);
int strata_fail(struct strata_error *err, const char *fmt, ...);
int strata_dimensions(int w, int h, struct strata_error *err);
float strata_srgb(float x);
int strata_read_exr(const char *path, struct strata_image *image,
		    struct strata_error *err);
struct strata_tensor {
	int width;
	int height;
	float *data;
};

int strata_tensor_alloc(struct strata_tensor *tensor, int w, int h,
			struct strata_error *err);
void strata_tensor_free(struct strata_tensor *tensor);
int strata_normalize(const struct strata_image *image, float exposure,
		     struct strata_tensor *tensor, struct strata_error *err);
int strata_pad(const struct strata_tensor *input, struct strata_tensor *padded,
	       struct strata_error *err);
int strata_fov_input(const struct strata_tensor *input,
		     struct strata_tensor *crop, struct strata_error *err);
float strata_focal(const float *angles, int w, int h);
int strata_postprocess(const float *prediction, int stride, int w, int h,
		       float focal, struct strata_result *result,
		       struct strata_error *err);
#endif

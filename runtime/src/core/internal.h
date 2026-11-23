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

struct strata_tensor {
	int width;
	int height;
	float *data;
};

int strata_fail(struct strata_error *err, const char *fmt, ...);
int strata_dimensions(int w, int h, struct strata_error *err);
double strata_milliseconds(void);
int strata_camera_validate(const struct strata_camera *camera, int resolved,
			   struct strata_error *err);
int strata_pose_validate(const struct strata_pose *pose,
			 struct strata_error *err);
int strata_deep_validate(const struct strata_deep_result *deep,
			 struct strata_error *err);
int strata_check_manifest(const char *model, struct strata_error *err);
int strata_read_exr(const char *path, struct strata_image *image,
		    struct strata_error *err);
float strata_srgb(float x);
int strata_padded_size(int w, int h, int *pw, int *ph,
		       struct strata_error *err);
size_t strata_prepare(const struct strata_image *image, double exposure,
		      const struct strata_tensor *padded);
int strata_fov_input(const struct strata_tensor *padded, int w, int h,
		     struct strata_tensor *crop, struct strata_error *err);
int strata_focal(const float *angles, int w, int h, float *focal,
		 struct strata_error *err);
int strata_postprocess(const float *prediction, int stride,
		       struct strata_result *result, struct strata_error *err);
int strata_atomic_target(const char *path, char **temporary,
			 struct strata_error *err);
int strata_atomic_finish(const char *path, char *temporary, int ret,
			 struct strata_error *err);
#endif

#include "strata.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int plane_test(void)
{
	float z[] = { 2, 2, 2, 2 };
	float expected[] = { -.5f, -.25f, 2, .5f, -.25f, 2,
			     -.5f, .25f,  2, .5f, .25f,	 2 };
	float world[] = { 10.25f, 19.5f, 32, 10.25f, 20.5f, 32,
			  9.75f,  19.5f, 32, 9.75f,  20.5f, 32 };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .frame.camera = { .fx_px = 2,
							 .fy_px = 4,
							 .cx_px = 1,
							 .cy_px = 1 },
				       .camera_z_metres = z };
	struct strata_pose pose = { { 0, -1, 0, 10, 1, 0, 0, 20, 0, 0, 1, 30, 0,
				      0, 0, 1 } };
	struct strata_positions positions = { 0 };
	struct strata_error err = { { 0 } };
	int ret = 1;

	if (strata_unproject_depth(&depth, NULL, NULL, &positions, &err))
		goto out;
	if (positions.point_count != 4 ||
	    positions.space != STRATA_SPACE_CAMERA ||
	    positions.layout != STRATA_POSITION_FLAT ||
	    memcmp(positions.xyz, expected, sizeof(expected)))
		goto out;
	for (size_t i = 0; i < 4; ++i) {
		if (positions.valid[i] != 1)
			goto out;
	}
	strata_positions_free(&positions);
	if (strata_unproject_depth(&depth, NULL, &pose, &positions, &err))
		goto out;
	if (positions.space != STRATA_SPACE_WORLD ||
	    memcmp(positions.xyz, world, sizeof(world)))
		goto out;
	ret = 0;
out:
	if (ret)
		fprintf(stderr,
			"synthetic camera/plane unprojection failed: %s\n",
			err.message);
	strata_positions_free(&positions);
	return ret;
}

static int deep_layers_test(void)
{
	uint32_t counts[] = { 2, 0 };
	float z[] = { 4, 2 }, rgba[] = { 1, 0, 0, 1, 0, 1, 0, 1 };
	float expected[] = { -1, -.5f, 4, -.5f, -.25f, 2 };
	struct strata_deep_result deep = { .width = 2,
					   .height = 1,
					   .frame.camera = { .fx_px = 2,
							     .fy_px = 4,
							     .cx_px = 1,
							     .cy_px = 1 },
					   .sample_count = 2,
					   .counts = counts,
					   .rgba = rgba,
					   .z = z,
					   .zback = z };
	struct strata_positions positions = { 0 };
	struct strata_layers layers = { 0 };
	struct strata_error err = { { 0 } };
	int ret = 1;

	if (strata_unproject_deep(&deep, NULL, NULL, &positions, &err))
		goto out;
	if (positions.layout != STRATA_POSITION_DEEP ||
	    positions.point_count != 2 ||
	    memcmp(positions.xyz, expected, sizeof(expected)))
		goto out;
	strata_positions_free(&positions);
	if (strata_deep_to_layers(&deep, 3, &layers, &err))
		goto out;
	layers.z[1] = 2;
	if (strata_unproject_layers(&layers, NULL, NULL, &positions, &err))
		goto out;
	if (positions.layout != STRATA_POSITION_LAYERS ||
	    positions.layer_count != 3 || positions.point_count != 6 ||
	    memcmp(positions.xyz, expected + 3, 3 * sizeof(float)) ||
	    memcmp(positions.xyz + 6, expected, 3 * sizeof(float)))
		goto out;
	for (size_t i = 0; i < 6; ++i) {
		if (positions.valid[i] != (i == 0 || i == 2))
			goto out;
	}
	ret = 0;
out:
	if (ret)
		fprintf(stderr, "deep/layer unprojection failed: %s\n",
			err.message);
	strata_positions_free(&positions);
	strata_layers_free(&layers);
	return ret;
}

static int invalid_depth_test(void)
{
	float z[] = { NAN, INFINITY, 0, -1 };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .frame.camera = { .fx_px = 2,
							 .fy_px = 4,
							 .cx_px = 1,
							 .cy_px = 1 },
				       .camera_z_metres = z };
	struct strata_positions positions = { 0 };
	struct strata_error err = { { 0 } };
	int ret = 1;

	if (strata_unproject_depth(&depth, NULL, NULL, &positions, &err))
		goto out;
	for (size_t i = 0; i < 4; ++i) {
		if (positions.valid[i] || positions.xyz[i * 3] != 0 ||
		    positions.xyz[i * 3 + 1] != 0 ||
		    positions.xyz[i * 3 + 2] != 0)
			goto out;
	}
	strata_positions_free(&positions);
	depth.frame.camera.fx_px = 0;
	if (strata_unproject_depth(&depth, NULL, NULL, &positions, &err) != -1)
		goto out;
	ret = 0;
out:
	strata_positions_free(&positions);
	return ret;
}

int main(void)
{
	return plane_test() || deep_layers_test() || invalid_depth_test();
}

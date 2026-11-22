#include "strata.h"
#include "expect.h"
#include <math.h>
#include <string.h>

static const struct strata_camera camera = { .fx_px = 2,
					     .fy_px = 4,
					     .cx_px = 1,
					     .cy_px = 1,
					     .focal_source =
						     STRATA_FOCAL_PROVIDED };

static void plane_test(void)
{
	float q[] = { .5f, .5f, .5f, .5f };
	float expected[] = { -.5f, -.25f, 2, .5f, -.25f, 2,
			     -.5f, .25f,  2, .5f, .25f,	 2 };
	float scaled[] = { -.5f, -.25f, 4, .5f, -.25f, 4,
			   -.5f, .25f,	4, .5f, .25f,  4 };
	struct strata_camera wider = { .fx_px = 4,
				       .fy_px = 8,
				       .cx_px = 1,
				       .cy_px = 1,
				       .focal_source = STRATA_FOCAL_PROVIDED };
	float world[] = { 10.25f, 19.5f, 32, 10.25f, 20.5f, 32,
			  9.75f,  19.5f, 32, 9.75f,  20.5f, 32 };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .frame.camera = camera,
				       .network_inverse_depth = q };
	struct strata_pose pose = { { 0, -1, 0, 10, 1, 0, 0, 20, 0, 0, 1, 30, 0,
				      0, 0, 1 } };
	struct strata_positions positions = { 0 };
	struct strata_error err = { { 0 } };

	EXPECT(strata_camera_z_metres(&depth, 0) == 2);
	EXPECT(strata_inverse_metres(&depth, 0) == .5f);
	if (EXPECT(!strata_unproject_depth(&depth, NULL, NULL, &positions,
					   &err)) &&
	    EXPECT(positions.point_count == 4)) {
		EXPECT(positions.space == STRATA_SPACE_CAMERA &&
		       positions.layout == STRATA_POSITION_FLAT);
		EXPECT(!memcmp(positions.xyz, expected, sizeof(expected)));
		for (size_t i = 0; i < 4; ++i)
			EXPECT(positions.valid[i] == 1);
	}
	if (EXPECT(!strata_unproject_depth(&depth, NULL, &pose, &positions,
					   &err)))
		EXPECT(positions.space == STRATA_SPACE_WORLD &&
		       !memcmp(positions.xyz, world, sizeof(world)));
	if (EXPECT(!strata_unproject_depth(&depth, &wider, NULL, &positions,
					   &err)))
		EXPECT(!memcmp(positions.xyz, scaled, sizeof(scaled)) &&
		       positions.frame.camera.fx_px == 4);
	strata_positions_free(&positions);
}

static void deep_layers_test(void)
{
	uint32_t counts[] = { 2, 0 };
	float z[] = { 4, 2 }, rgba[] = { 1, 0, 0, 1, 0, 1, 0, 1 };
	float expected[] = { -1, -.5f, 4, -.5f, -.25f, 2 };
	struct strata_deep_result deep = { .width = 2,
					   .height = 1,
					   .frame.camera = camera,
					   .sample_count = 2,
					   .counts = counts,
					   .rgba = rgba,
					   .z = z,
					   .zback = z };
	struct strata_positions positions = { 0 };
	struct strata_layers layers = { 0 };
	struct strata_error err = { { 0 } };

	if (EXPECT(!strata_unproject_deep(&deep, NULL, NULL, &positions, &err)))
		EXPECT(positions.layout == STRATA_POSITION_DEEP &&
		       positions.point_count == 2 &&
		       !memcmp(positions.xyz, expected, sizeof(expected)));
	if (!EXPECT(!strata_deep_to_layers(&deep, 3, &layers, &err)))
		return;
	layers.z[1] = 2;
	if (EXPECT(!strata_unproject_layers(&layers, NULL, NULL, &positions,
					    &err)) &&
	    EXPECT(positions.layout == STRATA_POSITION_LAYERS &&
		   positions.layer_count == 3 && positions.point_count == 6)) {
		EXPECT(!memcmp(positions.xyz, expected + 3, 3 * sizeof(float)));
		EXPECT(!memcmp(positions.xyz + 6, expected, 3 * sizeof(float)));
		for (size_t i = 0; i < 6; ++i)
			EXPECT(positions.valid[i] == (i == 0 || i == 2));
	}
	strata_positions_free(&positions);
	strata_layers_free(&layers);
}

static void invalid_test(void)
{
	float q[] = { NAN, 0, INFINITY, -1 };
	struct strata_result depth = { .width = 2,
				       .height = 2,
				       .frame.camera = camera,
				       .network_inverse_depth = q };
	struct strata_positions positions = { 0 };
	struct strata_error err = { { 0 } };

	if (EXPECT(!strata_unproject_depth(&depth, NULL, NULL, &positions,
					   &err))) {
		for (size_t i = 0; i < 4; ++i)
			EXPECT(!positions.valid[i] && !positions.xyz[i * 3] &&
			       !positions.xyz[i * 3 + 1] &&
			       !positions.xyz[i * 3 + 2]);
	}
	depth.frame.camera.fx_px = 0;
	EXPECT(strata_unproject_depth(&depth, NULL, NULL, &positions, &err) ==
	       -1);
	EXPECT(!positions.xyz && !positions.valid && !positions.point_count);
	depth.frame.camera.fx_px = 2;
	depth.frame.camera.focal_source = STRATA_FOCAL_UNSET;
	EXPECT(strata_unproject_depth(&depth, NULL, NULL, &positions, &err) ==
	       -1);
	EXPECT(strstr(err.message, "focal source") != NULL);
}

int main(void)
{
	plane_test();
	deep_layers_test();
	invalid_test();
	return failures != 0;
}

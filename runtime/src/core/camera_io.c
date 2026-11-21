#include "internal.h"
#include <json-c/json.h>
#include <float.h>

static struct json_object *read_camera(const char *path,
				       struct strata_error *err)
{
	char data[4096];
	FILE *file = fopen(path, "rb");
	struct json_tokener *tokener = NULL;
	struct json_object *object = NULL;
	size_t count;

	if (!file) {
		strata_fail(err, "cannot open camera: %s", path);
		return NULL;
	}
	count = fread(data, 1, sizeof(data), file);
	if (ferror(file) || count == sizeof(data)) {
		strata_fail(err, "camera JSON must be smaller than 4096 bytes");
		goto out;
	}
	tokener = json_tokener_new_ex(8);
	if (!tokener) {
		strata_fail(err, "out of memory");
		goto out;
	}
	json_tokener_set_flags(tokener, JSON_TOKENER_STRICT |
						JSON_TOKENER_VALIDATE_UTF8);
	object = json_tokener_parse_ex(tokener, data, (int)count);
	if (json_tokener_get_error(tokener) != json_tokener_success ||
	    json_tokener_get_parse_end(tokener) != count ||
	    !json_object_is_type(object, json_type_object)) {
		strata_fail(err, "invalid camera JSON");
		json_object_put(object);
		object = NULL;
	}
out:
	if (tokener)
		json_tokener_free(tokener);
	fclose(file);
	return object;
}

static int number(struct json_object *object, float *value,
		  struct strata_error *err)
{
	double decoded;

	if (!json_object_is_type(object, json_type_double) &&
	    !json_object_is_type(object, json_type_int))
		return strata_fail(err, "camera fields must be numeric");
	decoded = json_object_get_double(object);
	if (!isfinite(decoded) || fabs(decoded) > FLT_MAX)
		return strata_fail(err, "camera field is not finite float32");
	*value = (float)decoded;
	return 0;
}

static int decode_camera(struct json_object *root, struct strata_camera *camera,
			 struct strata_pose *pose, int *has_pose,
			 struct strata_error *err)
{
	const char *names[] = { "fx_px", "fy_px",	    "cx_px",
				"cy_px", "sensor_width_mm", "sensor_height_mm" };
	float *values[] = {
		&camera->fx_px,		  &camera->fy_px,
		&camera->cx_px,		  &camera->cy_px,
		&camera->sensor_width_mm, &camera->sensor_height_mm
	};
	struct json_object *field;
	int known = 1, intrinsics = 0;

	if (!json_object_object_get_ex(root, "version", &field) ||
	    !json_object_is_type(field, json_type_int) ||
	    json_object_get_int(field) != 1)
		return strata_fail(err, "camera JSON requires version 1");
	for (size_t i = 0; i < 6; ++i) {
		if (!json_object_object_get_ex(root, names[i], &field))
			continue;
		++known;
		if (i < 4)
			++intrinsics;
		if (number(field, values[i], err))
			return -1;
	}
	if ((intrinsics && intrinsics != 4) || camera->sensor_width_mm < 0 ||
	    camera->sensor_height_mm < 0)
		return strata_fail(
			err, "camera requires all four intrinsics or none");
	if (intrinsics && strata_camera_validate(camera, err))
		return -1;
	camera->focal_source = intrinsics ? STRATA_FOCAL_PROVIDED :
					    STRATA_FOCAL_ESTIMATED;
	if (json_object_object_get_ex(root, "camera_to_world", &field)) {
		++known;
		if (!json_object_is_type(field, json_type_array) ||
		    json_object_array_length(field) != 16)
			return strata_fail(
				err,
				"camera_to_world requires 16 row-major numbers");
		for (size_t i = 0; i < 16; ++i) {
			if (number(json_object_array_get_idx(field, i),
				   &pose->camera_to_world[i], err))
				return -1;
		}
		if (strata_pose_validate(pose, err))
			return -1;
		*has_pose = 1;
	}
	if (known != json_object_object_length(root))
		return strata_fail(err, "unknown camera JSON field");
	return 0;
}

int strata_camera_read(const char *path, struct strata_camera *camera,
		       struct strata_pose *pose, int *has_pose,
		       struct strata_error *err)
{
	struct json_object *object;
	int ret;

	if (!path || !camera || !pose || !has_pose)
		return strata_fail(err, "missing camera JSON input/output");
	memset(camera, 0, sizeof(*camera));
	memset(pose, 0, sizeof(*pose));
	*has_pose = 0;
	object = read_camera(path, err);
	if (!object)
		return -1;
	ret = decode_camera(object, camera, pose, has_pose, err);
	json_object_put(object);
	return ret;
}

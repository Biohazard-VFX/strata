#include "internal.h"
#include <ctype.h>
#include <json-c/json.h>

int strata_check_manifest(const char *model, struct strata_error *err)
{
	char buffer[16384];
	char *path = strata_manifest_path(model);
	FILE *file = NULL;
	struct json_tokener *parser = NULL;
	struct json_object *actual = NULL, *expected = NULL;
	size_t count, parsed;
	int ret = -1;

	if (!path)
		return strata_fail(err, "out of memory");
	file = fopen(path, "rb");
	if (!file) {
		strata_fail(err, "model contract missing: %s", path);
		goto out;
	}
	count = fread(buffer, 1, sizeof(buffer) - 1, file);
	if (ferror(file) || fgetc(file) != EOF) {
		strata_fail(err, "cannot read model contract: %s", path);
		goto out;
	}
	buffer[count] = '\0';
	parser = json_tokener_new();
	if (!parser) {
		strata_fail(err, "out of memory");
		goto out;
	}
	json_tokener_set_flags(parser, JSON_TOKENER_STRICT |
					       JSON_TOKENER_VALIDATE_UTF8);
	actual = json_tokener_parse_ex(parser, buffer, (int)count + 1);
	expected = json_tokener_parse(STRATA_CONTRACT_JSON);
	parsed = json_tokener_get_parse_end(parser);
	while (parsed < count && isspace((unsigned char)buffer[parsed]))
		++parsed;
	if (!actual || !expected ||
	    json_tokener_get_error(parser) != json_tokener_success ||
	    parsed < count || !json_object_equal(actual, expected)) {
		strata_fail(err, "incompatible model contract: %s", path);
		goto out;
	}
	ret = 0;
out:
	if (file)
		fclose(file);
	if (parser)
		json_tokener_free(parser);
	json_object_put(actual);
	json_object_put(expected);
	free(path);
	return ret;
}

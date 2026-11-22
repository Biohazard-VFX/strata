#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum message { MODEL, GRAPH, NODE, ATTRIBUTE, TENSOR, SPARSE, FUNCTION };

struct nesting {
	enum message parent;
	uint32_t field;
	enum message child;
};

static const struct nesting nested[] = {
	{ MODEL, 7, GRAPH },	   { MODEL, 25, FUNCTION },
	{ GRAPH, 1, NODE },	   { GRAPH, 5, TENSOR },
	{ GRAPH, 15, SPARSE },	   { NODE, 5, ATTRIBUTE },
	{ ATTRIBUTE, 5, TENSOR },  { ATTRIBUTE, 6, GRAPH },
	{ ATTRIBUTE, 10, TENSOR }, { ATTRIBUTE, 11, GRAPH },
	{ ATTRIBUTE, 22, SPARSE }, { ATTRIBUTE, 23, SPARSE },
	{ SPARSE, 1, TENSOR },	   { SPARSE, 2, TENSOR },
	{ FUNCTION, 7, NODE },	   { FUNCTION, 11, ATTRIBUTE },
};

struct span {
	const uint8_t *at;
	const uint8_t *end;
};

struct scan {
	struct strata_model_files *files;
	const char *model;
	size_t directory;
	struct strata_error *err;
};

void strata_model_files_free(struct strata_model_files *files)
{
	if (!files)
		return;
	for (size_t i = 0; i < files->count; ++i)
		free(files->paths[i]);
	free(files->paths);
	memset(files, 0, sizeof(*files));
}

static int append(struct strata_model_files *files, char *path,
		  struct strata_error *err)
{
	char **paths;

	if (!path)
		return strata_fail(err, "out of memory");
	for (size_t i = 0; i < files->count; ++i) {
		if (!strcmp(files->paths[i], path)) {
			free(path);
			return 0;
		}
	}
	paths = realloc(files->paths, (files->count + 1) * sizeof(*paths));
	if (!paths) {
		free(path);
		return strata_fail(err, "out of memory");
	}
	files->paths = paths;
	files->paths[files->count++] = path;
	return 0;
}

static int varint(struct span *span, uint64_t *value)
{
	*value = 0;
	for (unsigned int shift = 0; shift < 64 && span->at < span->end;
	     shift += 7) {
		uint8_t byte = *span->at++;

		*value |= (uint64_t)(byte & 0x7f) << shift;
		if (!(byte & 0x80))
			return 0;
	}
	return -1;
}

static int next_field(struct span *span, uint32_t *field, int *message,
		      struct span *payload)
{
	uint64_t key, length;

	if (varint(span, &key))
		return -1;
	*field = (uint32_t)(key >> 3);
	*message = (key & 7) == 2;
	switch (key & 7) {
	case 0:
		return varint(span, &length);
	case 1:
		length = 8;
		break;
	case 2:
		if (varint(span, &length))
			return -1;
		break;
	case 5:
		length = 4;
		break;
	default:
		return -1;
	}
	if (length > (uint64_t)(span->end - span->at))
		return -1;
	payload->at = span->at;
	payload->end = span->at + length;
	span->at = payload->end;
	return 0;
}

static int location(struct scan *scan, struct span entry)
{
	struct span key = { 0 }, value = { 0 }, payload;
	uint32_t field;
	size_t length, prefix;
	int message;
	char *path;

	while (entry.at < entry.end) {
		if (next_field(&entry, &field, &message, &payload))
			return strata_fail(scan->err,
					   "malformed ONNX model: %s",
					   scan->model);
		if (message && field == 1)
			key = payload;
		else if (message && field == 2)
			value = payload;
	}
	if (key.end - key.at != 8 || memcmp(key.at, "location", 8))
		return 0;
	length = (size_t)(value.end - value.at);
	if (!length || memchr(value.at, '\0', length))
		return strata_fail(scan->err,
				   "invalid external data location: %s",
				   scan->model);
	prefix = value.at[0] == '/' ? 0 : scan->directory;
	path = malloc(prefix + length + 1);
	if (path) {
		memcpy(path, scan->model, prefix);
		memcpy(path + prefix, value.at, length);
		path[prefix + length] = '\0';
	}
	return append(scan->files, path, scan->err);
}

static int walk(struct scan *scan, struct span span, enum message kind,
		int depth)
{
	struct span payload;
	uint32_t field;
	int message;

	if (depth > 64)
		return strata_fail(scan->err, "ONNX model nests too deeply: %s",
				   scan->model);
	while (span.at < span.end) {
		if (next_field(&span, &field, &message, &payload))
			return strata_fail(scan->err,
					   "malformed ONNX model: %s",
					   scan->model);
		if (!message)
			continue;
		if (kind == TENSOR && field == 13 && location(scan, payload))
			return -1;
		for (size_t i = 0; i < sizeof(nested) / sizeof(nested[0]);
		     ++i) {
			if (nested[i].parent == kind &&
			    nested[i].field == field &&
			    walk(scan, payload, nested[i].child, depth + 1))
				return -1;
		}
	}
	return 0;
}

static int external_data(struct strata_model_files *files, const char *model,
			 struct strata_error *err)
{
	const char *slash = strrchr(model, '/');
	struct scan scan = { files, model,
			     slash ? (size_t)(slash - model) + 1 : 0, err };
	struct stat status;
	uint8_t *bytes = MAP_FAILED;
	size_t size = 0;
	int fd = open(model, O_RDONLY | O_CLOEXEC), ret;

	if (fd < 0 && errno == ENOENT)
		return 0;
	if (fd >= 0 && !fstat(fd, &status) && S_ISREG(status.st_mode)) {
		size = (size_t)status.st_size;
		bytes = size ? mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0) :
			       NULL;
	}
	if (fd >= 0)
		close(fd);
	if (bytes == MAP_FAILED)
		return strata_fail(err, "cannot read model: %s", model);
	if (!size)
		return 0;
	ret = walk(&scan, (struct span){ bytes, bytes + size }, MODEL, 0);
	munmap(bytes, size);
	return ret;
}

int strata_model_files(const char *model_path, struct strata_model_files *files,
		       struct strata_error *err)
{
	char *fov;
	int ret;

	if (!files)
		return strata_fail(err, "missing model file output");
	strata_model_files_free(files);
	if (!model_path)
		return strata_fail(err, "missing model path");
	fov = strata_fov_path(model_path);
	if (!fov)
		return strata_fail(err, "out of memory");
	ret = append(files, strdup(model_path), err) ||
	      append(files, strata_manifest_path(model_path), err) ||
	      append(files, strdup(fov), err) ||
	      external_data(files, model_path, err) ||
	      external_data(files, fov, err);
	free(fov);
	if (ret)
		strata_model_files_free(files);
	return ret ? -1 : 0;
}

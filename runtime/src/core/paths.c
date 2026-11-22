#define _XOPEN_SOURCE 700
#include "internal.h"
#include <sys/stat.h>

struct identity {
	char *path;
	struct stat status;
	int exists;
};

static char *resolve_missing(const char *path)
{
	char *copy = strdup(path), *slash, *parent = NULL, *joined = NULL;
	const char *directory = ".", *name = copy;

	if (!copy)
		return NULL;
	slash = strrchr(copy, '/');
	if (slash) {
		*slash = '\0';
		directory = slash == copy ? "/" : copy;
		name = slash + 1;
	}
	if (*name && strcmp(name, ".") && strcmp(name, ".."))
		parent = realpath(directory, NULL);
	if (parent) {
		joined = malloc(strlen(parent) + strlen(name) + 2);
		if (joined)
			sprintf(joined, "%s/%s",
				strcmp(parent, "/") ? parent : "", name);
	}
	free(parent);
	free(copy);
	return joined;
}

static int identify(const char *path, struct identity *out,
		    struct strata_error *err)
{
	out->path = realpath(path, NULL);
	if (!out->path)
		out->path = resolve_missing(path);
	if (!out->path)
		return strata_fail(err, "cannot resolve path: %s", path);
	out->exists = !stat(out->path, &out->status);
	if (out->exists && S_ISDIR(out->status.st_mode))
		return strata_fail(err, "path is a directory: %s", path);
	return 0;
}

int strata_same_file(const char *a, const char *b, int *same,
		     struct strata_error *err)
{
	struct identity first = { 0 }, second = { 0 };
	int ret = -1;

	if (!a || !b || !same)
		return strata_fail(err, "missing path comparison arguments");
	*same = 0;
	if (identify(a, &first, err) || identify(b, &second, err))
		goto out;
	*same = !strcmp(first.path, second.path) ||
		(first.exists && second.exists &&
		 first.status.st_dev == second.status.st_dev &&
		 first.status.st_ino == second.status.st_ino);
	ret = 0;
out:
	free(first.path);
	free(second.path);
	return ret;
}

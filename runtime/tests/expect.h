#ifndef STRATA_TEST_EXPECT_H
#define STRATA_TEST_EXPECT_H
#include <stdio.h>

static int failures;

static inline int expect(int ok, const char *what, const char *file, int line)
{
	if (!ok) {
		fprintf(stderr, "%s:%d: %s\n", file, line, what);
		++failures;
	}
	return ok;
}

#define EXPECT(condition) expect(!!(condition), #condition, __FILE__, __LINE__)

static inline int zeroed(const void *object, size_t size)
{
	const unsigned char *bytes = object;

	for (size_t i = 0; i < size; ++i) {
		if (bytes[i])
			return 0;
	}
	return 1;
}

#define EXPECT_ZERO(object) EXPECT(zeroed(&(object), sizeof(object)))
#endif

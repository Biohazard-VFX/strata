#ifndef STRATA_TEST_EXPECT_H
#define STRATA_TEST_EXPECT_H
#include <stdio.h>

static int failures;

static int expect(int ok, const char *what, const char *file, int line)
{
	if (!ok) {
		fprintf(stderr, "%s:%d: %s\n", file, line, what);
		++failures;
	}
	return ok;
}

#define EXPECT(condition) expect(!!(condition), #condition, __FILE__, __LINE__)
#endif

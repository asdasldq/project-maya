// tests/glm_prof_test.cpp - the CPU-lane counter is a sum across parts, and the max is reported beside it.
#include "strata/core/glm_prof.hpp"

#include <cstdio>

static int g_fail = 0;
#define CHECK(c)                                                                                 \
    do {                                                                                         \
        if (!(c)) {                                                                              \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                    \
            ++g_fail;                                                                            \
        }                                                                                        \
    } while (0)

int main() {
    namespace gp = strata::core::glmprof;
    const double ms[] = {10.0, 29.0, 4.0, 2.0};
    const gp::CpuRollup r = gp::rollup(ms, 4);
    CHECK(r.parts == 4);
    CHECK(r.sum_ms == 45.0);
    CHECK(r.max_ms == 29.0);
    CHECK(r.max_i == 1);
    const gp::CpuRollup e = gp::rollup(nullptr, 4);
    CHECK(e.parts == 0 && e.max_i < 0 && e.sum_ms == 0.0);
    const gp::CpuRollup z = gp::rollup(ms, 0);
    CHECK(z.parts == 0);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("glm_prof_test ok\n");
    return 0;
}

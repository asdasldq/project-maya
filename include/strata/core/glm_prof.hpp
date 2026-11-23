// include/strata/core/glm_prof.hpp - how the decode's CPU-lane counter is reduced.
//
// glm stat's "cpu lane ms/tok" adds every part's own compute time and divides by output tokens.
// Parts run together, so that sum is not the time the GPUs waited. max_ms is one part's time, a
// lower bound on the lane's wall only when that part's compute is on the critical path.
#pragma once

#include <cstddef>

namespace strata::core::glmprof {

struct CpuRollup {
    double sum_ms = 0;
    double max_ms = 0;
    int parts = 0;
    int max_i = -1;
};

inline CpuRollup rollup(const double* ms, int n) {
    CpuRollup r;
    if (ms == nullptr || n <= 0) return r;
    for (int i = 0; i < n; ++i) {
        r.sum_ms += ms[i];
        if (r.parts == 0 || ms[i] > r.max_ms) {
            r.max_ms = ms[i];
            r.max_i = i;
        }
        ++r.parts;
    }
    return r;
}

}  // namespace strata::core::glmprof

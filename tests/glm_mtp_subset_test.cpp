// tests/glm_mtp_subset_test.cpp - draft-vocabulary rule shared by decode_spec and the pipe. No GPU.
#include "strata/core/glm_mtp_subset.hpp"

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
    using strata::core::mtp_draft_subset;
    // decode_spec passes caller_sub false. 4 GPUs, answer still inside the draft vocabulary.
    CHECK(mtp_draft_subset(false, false, false, 4, 0.0, 1000));
    CHECK(mtp_draft_subset(false, false, false, 4, 0.019, 1000));
    // Once the emitted-token EMA crosses 2%, the draft uses the whole head. Forcing caller_sub
    // keeps the subset, which a full-head verify cannot accept.
    CHECK(!mtp_draft_subset(false, false, false, 4, 0.02, 1000));
    CHECK(!mtp_draft_subset(false, false, false, 4, 0.5, 1000));
    CHECK(mtp_draft_subset(true, false, false, 4, 0.5, 1000));
    // No draft vocabulary, a 2-GPU split, or an explicit full head: the whole head.
    CHECK(!mtp_draft_subset(false, false, false, 4, 0.0, 0));
    CHECK(!mtp_draft_subset(true, false, false, 4, 0.0, 0));
    CHECK(!mtp_draft_subset(false, false, false, 2, 0.0, 1000));
    CHECK(!mtp_draft_subset(false, true, false, 4, 0.0, 1000));
    CHECK(!mtp_draft_subset(false, false, true, 4, 0.0, 1000));
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("glm_mtp_subset_test ok\n");
    return 0;
}

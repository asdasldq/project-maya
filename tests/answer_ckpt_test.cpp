// tests/answer_ckpt_test.cpp - the generated-token checkpoint length. No GPU.
#include "strata/core/answer_ckpt.hpp"

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
    using strata::core::answer_ckpt_len;
    // 76-token prompt, 256 written, the last one not fed: keep 331.
    CHECK(answer_ckpt_len(76, 256, 331) == 331);
    // The prompt snapshot's position is not an answer boundary.
    CHECK(answer_ckpt_len(76, 256, 75) == 0);
    CHECK(answer_ckpt_len(76, 256, 76) == 0);
    // A position at or past every written token is not the "last token still unfed" state.
    CHECK(answer_ckpt_len(76, 256, 332) == 0);
    CHECK(answer_ckpt_len(76, 0, 76) == 0);
    CHECK(answer_ckpt_len(76, 1, 76) == 0);
    CHECK(answer_ckpt_len(18, 10, 27) == 27);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("answer_ckpt_test ok\n");
    return 0;
}

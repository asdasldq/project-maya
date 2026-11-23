// tests/prefill_layout_test.cpp - layout-miss decision before any lend. No GPU.
#include "strata/core/prefill_layout.hpp"

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
    using strata::core::PrefillLayoutStep;
    using strata::core::prefill_layout_step;
    constexpr int kFloor = 256;
    // The measured 22-token tail asks for a 64-token chunk. It runs when that layout fits.
    CHECK(prefill_layout_step(true, 64, kFloor) == PrefillLayoutStep::Run);
    // The same chunk, when it does not fit, has no smaller batched layout. The tokens stay on the token path.
    CHECK(prefill_layout_step(false, 64, kFloor) == PrefillLayoutStep::TokenPath);
    CHECK(prefill_layout_step(false, 128, kFloor) == PrefillLayoutStep::TokenPath);
    CHECK(prefill_layout_step(false, 1, kFloor) == PrefillLayoutStep::TokenPath);
    // A longer prompt that does not fit still shrinks, down to the floor.
    CHECK(prefill_layout_step(false, 512, kFloor) == PrefillLayoutStep::Shrink);
    CHECK(prefill_layout_step(false, 320, kFloor) == PrefillLayoutStep::Shrink);
    CHECK(prefill_layout_step(true, 4096, kFloor) == PrefillLayoutStep::Run);
    // At the floor a miss is still an error. That is the long prompt's memory limit, not a silent serial read.
    CHECK(prefill_layout_step(false, kFloor, kFloor) == PrefillLayoutStep::Error);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("prefill_layout_test ok\n");
    return 0;
}

// include/strata/core/answer_ckpt.hpp - how much of a finished answer the conversation checkpoint may keep.
//
// No CUDA. The GLM serve loop feeds every written token except the last, and position() is the
// number of tokens fed. That prefix is what a later prompt can reuse when it sends the same ids
// back. The prompt snapshot (before the assistant turn) stays as it is.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::core {

// 0 when there is nothing past the prompt, or the position is not strictly inside
// [prompt + 1, prompt + emitted).
inline size_t answer_ckpt_len(size_t prompt, size_t emitted, int64_t pos) {
    if (emitted == 0 || pos <= (int64_t) prompt) return 0;
    const int64_t full = (int64_t) prompt + (int64_t) emitted;
    if (pos >= full) return 0;
    return (size_t) pos;
}

}  // namespace strata::core

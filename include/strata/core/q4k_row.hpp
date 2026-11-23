// include/strata/core/q4k_row.hpp - one Q4_K value, host or device.
//
// ggml's dequantize_row_q4_K (ggml-quants.c at 3cf03257): a 144-byte super-block holds fp16 d,
// fp16 dmin, 12 scale bytes, and 128 quant bytes. Element e is (d * sc) * nibble - (dmin * mn),
// products left to right. sc and mn are get_scale_min_k4. The chain's embed_tok uses this, and
// its setup refuses the chain unless the row matches the host dequantizer bit for bit.
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define STRATA_Q4K_HD __host__ __device__
#else
#define STRATA_Q4K_HD
#endif

namespace strata::core {

STRATA_Q4K_HD inline void q4k_scale_min(int j, const uint8_t* q, int& sc, int& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// block is 144 bytes. e is 0..255 inside that super-block. d and dmin are the block's fp16 scales
// already widened, so the caller controls the conversion.
STRATA_Q4K_HD inline float q4k_elem(const uint8_t* block, int e, float d, float dmin) {
    const int group = e >> 6;
    const int within = e & 63;
    const int hi = within >> 5;
    const int l = within & 31;
    const int is = group * 2 + hi;
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16 + group * 32;
    int sc = 0, mn = 0;
    q4k_scale_min(is, scales, sc, mn);
    const int nibble = hi ? (qs[l] >> 4) : (qs[l] & 0x0F);
    const float d1 = d * (float) sc;
    const float m1 = dmin * (float) mn;
    return d1 * (float) nibble - m1;
}

}  // namespace strata::core

#undef STRATA_Q4K_HD

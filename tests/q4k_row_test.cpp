// tests/q4k_row_test.cpp - Q4_K element formula against the transcribed ggml row. No GPU.
#include "strata/artifact/dequant.hpp"
#include "strata/core/q4k_row.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

static int g_fail = 0;

static void check_block(const uint8_t* block) {
    float row[256];
    strata::dequantize_q4_K(block, row);
    const float d = strata::fp16_to_fp32(strata::read_u16(block));
    const float dmin = strata::fp16_to_fp32(strata::read_u16(block + 2));
    for (int e = 0; e < 256; ++e) {
        const float got = strata::core::q4k_elem(block, e, d, dmin);
        if (std::memcmp(&got, &row[e], sizeof(float)) != 0) {
            std::fprintf(stderr, "FAIL e=%d got=%a row=%a\n", e, got, row[e]);
            ++g_fail;
            return;
        }
    }
}

int main() {
    uint8_t block[144];
    for (int seed = 1; seed <= 7; ++seed) {
        for (int i = 0; i < 144; ++i) block[i] = (uint8_t) (i * 17 + seed * 29);
        // A finite fp16 scale pair. 0x3c00 is 1.0, 0x3800 is 0.5.
        block[0] = 0x00;
        block[1] = 0x3c;
        block[2] = (uint8_t) (0x00 + seed);
        block[3] = 0x38;
        check_block(block);
    }
    std::memset(block, 0, sizeof(block));
    block[1] = 0x3c;
    block[3] = 0x3c;
    for (int i = 4; i < 16; ++i) block[i] = (uint8_t) (0xFF - i);
    for (int i = 16; i < 144; ++i) block[i] = (uint8_t) (i * 3);
    check_block(block);
    // Scale indexes 4..7 use the packed high bits. Force those bytes.
    std::memset(block, 0, sizeof(block));
    block[1] = 0x40;
    block[3] = 0x38;
    block[4] = 0xC3;
    block[5] = 0x15;
    block[8] = 0xA6;
    block[9] = 0x5A;
    block[12] = 0x3F;
    block[13] = 0xE1;
    for (int i = 16; i < 144; ++i) block[i] = (uint8_t) (0xF0 ^ i);
    check_block(block);
    if (g_fail != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    std::printf("q4k_row_test ok\n");
    return 0;
}

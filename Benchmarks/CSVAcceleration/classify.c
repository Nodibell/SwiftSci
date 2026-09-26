#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t comma, lf, quote, cr; } Marks;
static inline uint32_t mask16(uint8x16_t v, uint8_t c) {
    static const uint8_t weights[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t w = vandq_u8(vceqq_u8(v, vdupq_n_u8(c)), vld1q_u8(weights));
    return (uint32_t)vaddv_u8(vget_low_u8(w)) | ((uint32_t)vaddv_u8(vget_high_u8(w)) << 8);
}
void classify_neon(const uint8_t *bytes, size_t length, Marks *out) {
    size_t block = 0, pos = 0;
    for (; pos + 32 <= length; pos += 32, block++) {
        uint8x16_t a = vld1q_u8(bytes + pos), b = vld1q_u8(bytes + pos + 16);
        out[block] = (Marks){mask16(a,44) | (mask16(b,44)<<16), mask16(a,10) | (mask16(b,10)<<16), mask16(a,34) | (mask16(b,34)<<16), mask16(a,13) | (mask16(b,13)<<16)};
    }
    if (pos < length) {
        Marks m = {0,0,0,0};
        for (uint32_t bit = 1; pos < length; pos++, bit <<= 1) {
            if (bytes[pos] == 44) m.comma |= bit;
            if (bytes[pos] == 10) m.lf |= bit;
            if (bytes[pos] == 34) m.quote |= bit;
            if (bytes[pos] == 13) m.cr |= bit;
        }
        out[block] = m;
    }
}

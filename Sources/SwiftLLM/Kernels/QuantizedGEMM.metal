#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

/// Quantized 4-bit (Q4_0) matrix-vector GEMV kernel.
/// Block size: 32 elements. Each block contains a 16-bit float scale and 16 bytes of packed 4-bit signed integers.
kernel void gemv_q4_0(
    device const float* inVec                  [[buffer(0)]],
    device const uchar* weights                [[buffer(1)]],
    device const half* scales                  [[buffer(2)]],
    device float* outVec                       [[buffer(3)]],
    constant uint& inFeatures                  [[buffer(4)]],
    constant uint& outFeatures                 [[buffer(5)]],
    uint thread_position_in_grid               [[thread_position_in_grid]]
) {
    uint row = thread_position_in_grid;
    if (row >= outFeatures) {
        return;
    }

    uint blocksPerRow = inFeatures / 32;
    uint blockOffset = row * blocksPerRow;
    uint weightByteOffset = blockOffset * 16;

    float accumulator = 0.0f;
    for (uint b = 0; b < blocksPerRow; ++b) {
        float scale = float(scales[blockOffset + b]);
        device const uchar* bWeights = weights + weightByteOffset + (b * 16);
        device const float* bIn = inVec + (b * 32);

        for (uint i = 0; i < 16; ++i) {
            uchar byteVal = bWeights[i];
            int w0 = int(byteVal & 0x0F) - 8;
            int w1 = int(byteVal >> 4) - 8;

            accumulator += (float(w0) * scale) * bIn[2 * i];
            accumulator += (float(w1) * scale) * bIn[2 * i + 1];
        }
    }

    outVec[row] = accumulator;
}

/// Quantized 8-bit (Q8_0) matrix-vector GEMV kernel.
/// Block size: 32 elements. Each block contains a 16-bit float scale and 32 bytes of 8-bit signed integers.
kernel void gemv_q8_0(
    device const float* inVec                  [[buffer(0)]],
    device const int8_t* weights               [[buffer(1)]],
    device const half* scales                  [[buffer(2)]],
    device float* outVec                       [[buffer(3)]],
    constant uint& inFeatures                  [[buffer(4)]],
    constant uint& outFeatures                 [[buffer(5)]],
    uint thread_position_in_grid               [[thread_position_in_grid]]
) {
    uint row = thread_position_in_grid;
    if (row >= outFeatures) {
        return;
    }

    uint blocksPerRow = inFeatures / 32;
    uint blockOffset = row * blocksPerRow;
    uint weightByteOffset = blockOffset * 32;

    float accumulator = 0.0f;
    for (uint b = 0; b < blocksPerRow; ++b) {
        float scale = float(scales[blockOffset + b]);
        device const int8_t* bWeights = weights + weightByteOffset + (b * 32);
        device const float* bIn = inVec + (b * 32);

        for (uint i = 0; i < 32; ++i) {
            accumulator += (float(bWeights[i]) * scale) * bIn[i];
        }
    }

    outVec[row] = accumulator;
}

#if defined(__HAVE_SIMDGROUP_MATRIX__)
/// SIMD-group accelerated 8x8 matrix multiplication on Apple Silicon GPU.
kernel void gemm_simdgroup_q4_0(
    device const half* A                       [[buffer(0)]],
    device const half* B                       [[buffer(1)]],
    device half* C                             [[buffer(2)]],
    constant uint& M                           [[buffer(3)]],
    constant uint& N                           [[buffer(4)]],
    constant uint& K                           [[buffer(5)]],
    uint2 threadgroup_position_in_grid         [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup        [[simdgroup_index_in_threadgroup]]
) {
    simdgroup_matrix<half, 8, 8> acc;
    acc = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);

    uint tileRow = threadgroup_position_in_grid.y * 8;
    uint tileCol = threadgroup_position_in_grid.x * 8;

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<half, 8, 8> a_mat;
        simdgroup_matrix<half, 8, 8> b_mat;
        simdgroup_load(a_mat, A + (tileRow * K) + k, K);
        simdgroup_load(b_mat, B + (k * N) + tileCol, N);
        simdgroup_multiply_accumulate(acc, a_mat, b_mat, acc);
    }

    if (tileRow < M && tileCol < N) {
        simdgroup_store(acc, C + (tileRow * N) + tileCol, N);
    }
}
#endif

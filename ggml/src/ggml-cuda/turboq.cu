// CUDA TurboQuant kernels for TBQ3_0 and TBQ4_0 KV cache types.
//
// TBQ blocks are 128 normalized values encoded with 3-bit or 4-bit Gaussian
// codebooks. KV-cache rotation is supplied by llama.cpp's shared attention
// rotation graph, so this backend must not apply a second transform.

#include "turboq.cuh"
#include "../ggml-turboq-tables.h"

// ---------------------------------------------------------------------------
// Dequantize kernels
//
// Each TBQ block contains 128 elements.
// Launch: nb CUDA blocks with 128 threads each.
// Each CUDA block handles one 128-element TBQ block:
//   1. Unpack indices from qs
//   2. Look up codebook values
//   3. Scale by the block norm
// ---------------------------------------------------------------------------

// Device codebooks (matching turboq_codebook_3bit / turboq_codebook_4bit)
__device__ static const float d_codebook_3bit[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1520f,
};

__device__ static const float d_codebook_4bit[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9424f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9424f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f,
};

// 3-bit boundaries for quantize path
__device__ static const float d_boundaries_3bit[7] = {
    -1.7480f, -1.0500f, -0.5006f, 0.0000f,
     0.5006f,  1.0500f,  1.7480f,
};

// 4-bit boundaries for quantize path
__device__ static const float d_boundaries_4bit[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f,
    -0.7996f, -0.5225f, -0.2583f,  0.0000f,
     0.2583f,  0.5225f,  0.7996f,  1.0993f,
     1.4371f,  1.8435f,  2.4008f,
};

// Unpack a 3-bit index from packed qs array
static __device__ __forceinline__ int unpack_3bit_index(const uint8_t * qs, int elem) {
    const int group = elem / 8;
    const int bit_offset = (elem % 8) * 3;
    const uint8_t * src = qs + group * 3;
    uint32_t bits = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
    return (bits >> bit_offset) & 0x7;
}

template<typename dst_t>
__global__ void dequantize_block_tbq3_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y, int64_t nb) {

    const int tbq_block = blockIdx.x;
    const int tid = threadIdx.x;          // 0..127

    if (tbq_block >= nb) return;

    const block_tbq3_0 * x = (const block_tbq3_0 *)vx;
    const float norm = __half2float(x[tbq_block].d);
    const float scale_down = 0.08838834764f;  // 1/sqrt(128)

    // Step 1: Unpack 3-bit index and look up codebook
    const int elem = tid;
    const int idx = unpack_3bit_index(x[tbq_block].qs, elem);

    y[tbq_block * 128 + tid] = (dst_t)(d_codebook_3bit[idx] * scale_down * norm);
}

template<typename dst_t>
__global__ void dequantize_block_tbq4_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y, int64_t nb) {

    const int tbq_block = blockIdx.x;
    const int tid = threadIdx.x;

    if (tbq_block >= nb) return;

    const block_tbq4_0 * x = (const block_tbq4_0 *)vx;
    const float norm = __half2float(x[tbq_block].d);
    const float scale_down = 0.08838834764f;

    // Step 1: Unpack 4-bit nibble and look up codebook
    const int elem = tid;
    uint8_t idx;
    if (elem % 2 == 0) {
        idx = x[tbq_block].qs[elem / 2] & 0x0F;
    } else {
        idx = (x[tbq_block].qs[elem / 2] >> 4) & 0x0F;
    }

    y[tbq_block * 128 + tid] = (dst_t)(d_codebook_4bit[idx] * scale_down * norm);
}

template<typename dst_t>
void dequantize_row_tbq3_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    const int64_t nb = k / 128;
    dequantize_block_tbq3_0_kernel<<<nb, 128, 0, stream>>>(vx, y, nb);
}

template<typename dst_t>
void dequantize_row_tbq4_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    const int64_t nb = k / 128;
    dequantize_block_tbq4_0_kernel<<<nb, 128, 0, stream>>>(vx, y, nb);
}

// Explicit template instantiations
template void dequantize_row_tbq3_0_cuda<float>(const void * vx, float * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq3_0_cuda<half>(const void * vx, half * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq4_0_cuda<float>(const void * vx, float * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq4_0_cuda<half>(const void * vx, half * y, int64_t k, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Quantize kernels (f32 -> TBQ for KV cache write path)
//
// One CUDA block computes the norm and packs one TBQ block.
// ---------------------------------------------------------------------------

// TBQ3_0 quantize: normalize, apply the 3-bit codebook, and pack.
__global__ void quantize_f32_tbq3_0_kernel(
    const float * __restrict__ x, void * __restrict__ vy, int64_t num_blocks) {

    const int tbq_block = blockIdx.x;
    const int tid = threadIdx.x;          // 0..127

    if (tbq_block >= num_blocks) return;

    block_tbq3_0 * y = (block_tbq3_0 *)vy;
    const float * src = x + tbq_block * 128;
    const float scale_up = 11.3137085f;  // sqrt(128)

    const float val = src[tid];
    float sum_sq = warp_reduce_sum(val * val);
    __shared__ float warp_sums[4];
    __shared__ float s_norm;
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = sum_sq;
    }
    __syncthreads();
    if (tid < 32) {
        float s = tid < 4 ? warp_sums[tid] : 0.0f;
        for (int offset = 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (tid == 0) {
            const float norm = sqrtf(s);
            s_norm = norm < 1e-10f ? 1e-10f : norm;
        }
    }
    __syncthreads();

    const float scaled = val / s_norm * scale_up;
    int idx = 7;
    #pragma unroll
    for (int b = 0; b < 7; b++) {
        if (scaled < d_boundaries_3bit[b]) {
            idx = b;
            break;
        }
    }

    // Step 4: Pack 3-bit indices
    // Group of 8 indices -> 3 bytes. Each thread contributes to its group.
    const int group = tid / 8;
    const int pos_in_group = tid % 8;
    const int bit_offset = pos_in_group * 3;

    // Use shared memory for atomic packing
    __shared__ uint32_t s_packed[16];  // 128/8 = 16 groups, each needs 24 bits
    if (tid < 16) {
        s_packed[tid] = 0;
    }
    __syncthreads();

    atomicOr(&s_packed[group], ((uint32_t)idx) << bit_offset);
    __syncthreads();

    // Step 5: Write packed bytes to output
    // Each sub-block writes 48 bytes (128 * 3 / 8)
    const int qs_offset = 0;  // 128*3/8
    if (pos_in_group < 3) {
        // 3 bytes per group, thread 0-2 in each group write one byte
        uint32_t packed = s_packed[group];
        y[tbq_block].qs[qs_offset + group * 3 + pos_in_group] =
            (uint8_t)((packed >> (pos_in_group * 8)) & 0xFF);
    }

    // First thread writes the norm
    if (tid == 0) {
        y[tbq_block].d = __float2half(s_norm);
    }
}

// TBQ4_0 quantize: normalize, apply the 4-bit codebook, and pack nibbles.
__global__ void quantize_f32_tbq4_0_kernel(
    const float * __restrict__ x, void * __restrict__ vy, int64_t num_blocks) {

    const int tbq_block = blockIdx.x;
    const int tid = threadIdx.x;

    if (tbq_block >= num_blocks) return;

    block_tbq4_0 * y = (block_tbq4_0 *)vy;
    const float * src = x + tbq_block * 128;
    const float scale_up = 11.3137085f;

    const float val = src[tid];
    float sum_sq = warp_reduce_sum(val * val);
    __shared__ float warp_sums[4];
    __shared__ float s_norm;
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = sum_sq;
    }
    __syncthreads();
    if (tid < 32) {
        float s = tid < 4 ? warp_sums[tid] : 0.0f;
        for (int offset = 2; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        }
        if (tid == 0) {
            const float norm = sqrtf(s);
            s_norm = norm < 1e-10f ? 1e-10f : norm;
        }
    }
    __syncthreads();

    const float scaled = val / s_norm * scale_up;
    int idx = 15;
    #pragma unroll
    for (int b = 0; b < 15; b++) {
        if (scaled < d_boundaries_4bit[b]) {
            idx = b;
            break;
        }
    }

    // Step 4: Pack 4-bit nibbles via shared memory
    // Even indices go in low nibble, odd in high nibble
    __shared__ uint8_t s_indices[128];
    s_indices[tid] = (uint8_t)idx;
    __syncthreads();

    // Each even-index thread packs a pair
    if (tid % 2 == 0) {
        const int qs_idx = tid / 2;
        y[tbq_block].qs[qs_idx] = s_indices[tid] | (s_indices[tid + 1] << 4);
    }

    // Write norm
    if (tid == 0) {
        y[tbq_block].d = __float2half(s_norm);
    }
}

// ---------------------------------------------------------------------------
// Copy dispatch functions (called from cpy.cu)
// ---------------------------------------------------------------------------

void ggml_cpy_f32_tbq3_0_cuda(
    const char * cx, char * cdst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    cudaStream_t stream) {

    (void)ne00; (void)ne01; (void)ne02; (void)nb00; (void)nb01; (void)nb02; (void)nb03;
    (void)ne10; (void)ne11; (void)ne12; (void)nb10; (void)nb11; (void)nb12; (void)nb13;

    GGML_ASSERT(ne % 128 == 0);
    const int64_t num_blocks = ne / 128;
    quantize_f32_tbq3_0_kernel<<<num_blocks, 128, 0, stream>>>(
        (const float *)cx, cdst, num_blocks);
}

void ggml_cpy_f32_tbq4_0_cuda(
    const char * cx, char * cdst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    cudaStream_t stream) {

    (void)ne00; (void)ne01; (void)ne02; (void)nb00; (void)nb01; (void)nb02; (void)nb03;
    (void)ne10; (void)ne11; (void)ne12; (void)nb10; (void)nb11; (void)nb12; (void)nb13;

    GGML_ASSERT(ne % 128 == 0);
    const int64_t num_blocks = ne / 128;
    quantize_f32_tbq4_0_kernel<<<num_blocks, 128, 0, stream>>>(
        (const float *)cx, cdst, num_blocks);
}

// ---------------------------------------------------------------------------
// Fused SET_ROWS kernels (no host-device sync needed)
//
// For KV cache writes: src0 is float data, src1 is row indices, dst is TBQ.
// Each row of ne00 elements is quantized and written to the destination
// row indicated by src1.
//
// Architecture: 128 threads per CUDA block, 1 block per source row.
// Phase 1: compute the per-TBQ-block L2 norm via warp reduction.
// Phase 2: normalize, quantize, and pack each 128-value TBQ block.
// ---------------------------------------------------------------------------

template<typename idx_t>
__global__ void set_rows_tbq3_0_kernel(
    const float * __restrict__ src0, const idx_t * __restrict__ src1, char * __restrict__ dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int64_t ne11, int64_t ne12) {

    // Each CUDA block handles one source row
    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;  // 0..127

    // Decompose row_idx into (i01, i02, i03)
    const int64_t total_rows = ne01 * ne02 * ne03;
    if (row_idx >= total_rows) return;

    const int64_t i01 = row_idx % ne01;
    const int64_t i02 = (row_idx / ne01) % ne02;
    const int64_t i03 = row_idx / (ne01 * ne02);

    // Source row pointer
    const float * src_row = src0 + i01 * s01 + i02 * s02 + i03 * s03;

    // Destination row index from src1
    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;
    const int64_t dst_row_idx = (int64_t)src1[i10 * s10 + i11 * s11 + i12 * s12];

    // Destination pointer
    char * dst_row = dst + dst_row_idx * nb1 + i02 * nb2 + i03 * nb3;

    // ne00 should be 128 (one TBQ block per row) — assert in host code
    // For simplicity, handle exactly 1 TBQ block (128 elements)
    const int64_t num_blocks_per_row = ne00 / 128;

    for (int64_t blk = 0; blk < num_blocks_per_row; blk++) {
        const float * blk_src = src_row + blk * 128;
        block_tbq3_0 * blk_dst = (block_tbq3_0 *)dst_row + blk;

        // Phase 1: Compute L2 norm (all 128 threads)
        float val = blk_src[tid];
        float sum_sq = val * val;

        sum_sq = warp_reduce_sum(sum_sq);

        __shared__ float warp_sums[4];
        if (tid % 32 == 0) warp_sums[tid / 32] = sum_sq;
        __syncthreads();

        __shared__ float s_norm;
        if (tid < 32) {
            float s = (tid < 4) ? warp_sums[tid] : 0.0f;
            for (int offset = 2; offset > 0; offset >>= 1) {
                s += __shfl_down_sync(0xFFFFFFFF, s, offset);
            }
            if (tid == 0) {
                float n = sqrtf(s);
                s_norm = (n < 1e-10f) ? 1e-10f : n;
            }
        }
        __syncthreads();

        // The shared attention graph already rotated this KV row.
        const float scaled = val / s_norm * 11.3137085f;
        int idx = 7;
        #pragma unroll
        for (int b = 0; b < 7; b++) {
            if (scaled < d_boundaries_3bit[b]) { idx = b; break; }
        }

        // Pack 3-bit
        const int group = tid / 8;
        const int pos_in_group = tid % 8;

        __shared__ uint32_t s_packed[16];
        if (tid < 16) s_packed[tid] = 0;
        __syncthreads();

        atomicOr(&s_packed[group], ((uint32_t)idx) << (pos_in_group * 3));
        __syncthreads();

        if (pos_in_group < 3) {
            uint32_t packed = s_packed[group];
            blk_dst->qs[group * 3 + pos_in_group] =
                (uint8_t)((packed >> (pos_in_group * 8)) & 0xFF);
        }

        if (tid == 0) {
            blk_dst->d = __float2half(s_norm);
        }
        __syncthreads();
    }
}

template<typename idx_t>
__global__ void set_rows_tbq4_0_kernel(
    const float * __restrict__ src0, const idx_t * __restrict__ src1, char * __restrict__ dst,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int64_t ne11, int64_t ne12) {

    const int row_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const int64_t total_rows = ne01 * ne02 * ne03;
    if (row_idx >= total_rows) return;

    const int64_t i01 = row_idx % ne01;
    const int64_t i02 = (row_idx / ne01) % ne02;
    const int64_t i03 = row_idx / (ne01 * ne02);

    const float * src_row = src0 + i01 * s01 + i02 * s02 + i03 * s03;

    const int64_t i10 = i01;
    const int64_t i11 = i02 % ne11;
    const int64_t i12 = i03 % ne12;
    const int64_t dst_row_idx = (int64_t)src1[i10 * s10 + i11 * s11 + i12 * s12];

    char * dst_row = dst + dst_row_idx * nb1 + i02 * nb2 + i03 * nb3;

    const int64_t num_blocks_per_row = ne00 / 128;

    for (int64_t blk = 0; blk < num_blocks_per_row; blk++) {
        const float * blk_src = src_row + blk * 128;
        block_tbq4_0 * blk_dst = (block_tbq4_0 *)dst_row + blk;

        // Phase 1: L2 norm
        float val = blk_src[tid];
        float sum_sq = val * val;

        sum_sq = warp_reduce_sum(sum_sq);

        __shared__ float warp_sums[4];
        if (tid % 32 == 0) warp_sums[tid / 32] = sum_sq;
        __syncthreads();

        __shared__ float s_norm;
        if (tid < 32) {
            float s = (tid < 4) ? warp_sums[tid] : 0.0f;
            for (int offset = 2; offset > 0; offset >>= 1) {
                s += __shfl_down_sync(0xFFFFFFFF, s, offset);
            }
            if (tid == 0) {
                float n = sqrtf(s);
                s_norm = (n < 1e-10f) ? 1e-10f : n;
            }
        }
        __syncthreads();

        const float scaled = val / s_norm * 11.3137085f;
        int idx = 15;
        #pragma unroll
        for (int b = 0; b < 15; b++) {
            if (scaled < d_boundaries_4bit[b]) { idx = b; break; }
        }

        // Pack 4-bit nibbles
        __shared__ uint8_t s_indices[128];
        s_indices[tid] = (uint8_t)idx;
        __syncthreads();

        if (tid % 2 == 0) {
            const int qs_idx = tid / 2;
            blk_dst->qs[qs_idx] = s_indices[tid] | (s_indices[tid + 1] << 4);
        }

        if (tid == 0) {
            blk_dst->d = __float2half(s_norm);
        }
        __syncthreads();
    }
}

// Host dispatch functions
template<typename idx_t>
void ggml_set_rows_tbq3_0_cuda(
    const float * src0_d, const idx_t * src1_d, char * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne11, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream) {

    GGML_ASSERT(ne00 % 128 == 0);

    const int64_t total_rows = ne01 * ne02 * ne03;
    if (total_rows == 0 || ne11 == 0 || ne12 == 0) {
        return;
    }
    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    set_rows_tbq3_0_kernel<<<total_rows, 128, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        s01, s02, s03, s10, s11, s12,
        nb1, nb2, nb3, ne11, ne12);
}

template<typename idx_t>
void ggml_set_rows_tbq4_0_cuda(
    const float * src0_d, const idx_t * src1_d, char * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t ne11, int64_t ne12,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream) {

    GGML_ASSERT(ne00 % 128 == 0);

    const int64_t total_rows = ne01 * ne02 * ne03;
    if (total_rows == 0 || ne11 == 0 || ne12 == 0) {
        return;
    }
    const int64_t s01 = nb01 / sizeof(float);
    const int64_t s02 = nb02 / sizeof(float);
    const int64_t s03 = nb03 / sizeof(float);
    const int64_t s10 = nb10 / sizeof(idx_t);
    const int64_t s11 = nb11 / sizeof(idx_t);
    const int64_t s12 = nb12 / sizeof(idx_t);

    set_rows_tbq4_0_kernel<<<total_rows, 128, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, ne01, ne02, ne03,
        s01, s02, s03, s10, s11, s12,
        nb1, nb2, nb3, ne11, ne12);
}

// Explicit template instantiations for SET_ROWS
template void ggml_set_rows_tbq3_0_cuda<int32_t>(const float*, const int32_t*, char*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void ggml_set_rows_tbq3_0_cuda<int64_t>(const float*, const int64_t*, char*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void ggml_set_rows_tbq4_0_cuda<int32_t>(const float*, const int32_t*, char*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);
template void ggml_set_rows_tbq4_0_cuda<int64_t>(const float*, const int64_t*, char*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, cudaStream_t);

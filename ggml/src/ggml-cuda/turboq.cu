// CUDA TurboQuant kernels for TBQ3_0 and TBQ4_0 KV cache types.
//
// TBQ blocks are 256 elements with a 128x128 Householder rotation applied
// in two sub-blocks of 128. The rotation matrix Q is generated once on the
// host (using the CPU turboq code) and uploaded to device global memory.

#include "turboq.cuh"
#include "../ggml-turboq.h"
#include "../ggml-turboq-tables.h"

#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Rotation matrix manager
// ---------------------------------------------------------------------------

static float * d_turboq_Q = nullptr;  // device rotation matrix (128x128 row-major)

void ggml_cuda_turboq_init(cudaStream_t stream) {
    if (d_turboq_Q != nullptr) {
        return;  // already initialized
    }

    const int64_t d = 128;  // TURBOQ_KV_DIM

    // Generate rotation matrix on host by rotating identity vectors
    float * h_Q = (float *)malloc(d * d * sizeof(float));
    GGML_ASSERT(h_Q != nullptr);

    float * in  = (float *)calloc(d, sizeof(float));
    float * out = (float *)malloc(d * sizeof(float));
    GGML_ASSERT(in != nullptr && out != nullptr);

    const uint64_t seed = turboq_seed_from_row(0);

    for (int64_t col = 0; col < d; col++) {
        memset(in, 0, d * sizeof(float));
        in[col] = 1.0f;
        turboq_rotate_forward(out, in, d, seed);
        // out = Q * e_col, so out is column col of Q
        // Store row-major: h_Q[row * d + col] = out[row]
        for (int64_t row = 0; row < d; row++) {
            h_Q[row * d + col] = out[row];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_turboq_Q, d * d * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_turboq_Q, h_Q, d * d * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    free(h_Q);
    free(in);
    free(out);
}

void ggml_cuda_turboq_free(void) {
    if (d_turboq_Q != nullptr) {
        CUDA_CHECK(cudaFree(d_turboq_Q));
        d_turboq_Q = nullptr;
    }
}

const float * ggml_cuda_turboq_get_rotation(void) {
    return d_turboq_Q;
}

// ---------------------------------------------------------------------------
// Dequantize kernels
//
// Each TBQ block = 256 elements = 2 sub-blocks of 128.
// Launch: nb*2 CUDA blocks with 128 threads each.
// Each CUDA block handles one 128-element sub-block:
//   1. Unpack indices from qs
//   2. Look up codebook values
//   3. Inverse rotation via shared memory matvec (Q^T * rotated)
//   4. Scale by norm
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
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const float * __restrict__ Q_rot, int64_t nb) {

    const int sub_block = blockIdx.x;
    const int tbq_block = sub_block / 2;
    const int sub_half  = sub_block % 2;  // 0 = first 128, 1 = second 128
    const int tid = threadIdx.x;          // 0..127

    if (tbq_block >= nb) return;

    const block_tbq3_0 * x = (const block_tbq3_0 *)vx;
    const float norm = __half2float(x[tbq_block].d);
    const float scale_down = 0.0625f;  // 1/sqrt(256) = 1/16

    // Step 1: Unpack 3-bit index and look up codebook
    const int elem = sub_half * 128 + tid;
    const int idx = unpack_3bit_index(x[tbq_block].qs, elem);

    __shared__ float s_rotated[128];
    s_rotated[tid] = d_codebook_3bit[idx] * scale_down;
    __syncthreads();

    // Step 2: Inverse rotation: output[tid] = dot(Q^T[tid, :], s_rotated[:])
    // Q_rot is 128x128 row-major. Q^T[tid, col] = Q_rot[col * 128 + tid]
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
        sum += Q_rot[j * 128 + tid] * s_rotated[j];
    }

    // Step 3: Scale by norm and write
    y[tbq_block * 256 + sub_half * 128 + tid] = (dst_t)(sum * norm);
}

template<typename dst_t>
__global__ void dequantize_block_tbq4_0_kernel(
    const void * __restrict__ vx, dst_t * __restrict__ y,
    const float * __restrict__ Q_rot, int64_t nb) {

    const int sub_block = blockIdx.x;
    const int tbq_block = sub_block / 2;
    const int sub_half  = sub_block % 2;
    const int tid = threadIdx.x;

    if (tbq_block >= nb) return;

    const block_tbq4_0 * x = (const block_tbq4_0 *)vx;
    const float norm = __half2float(x[tbq_block].d);
    const float scale_down = 0.0625f;

    // Step 1: Unpack 4-bit nibble and look up codebook
    const int elem = sub_half * 128 + tid;
    uint8_t idx;
    if (elem % 2 == 0) {
        idx = x[tbq_block].qs[elem / 2] & 0x0F;
    } else {
        idx = (x[tbq_block].qs[elem / 2] >> 4) & 0x0F;
    }

    __shared__ float s_rotated[128];
    s_rotated[tid] = d_codebook_4bit[idx] * scale_down;
    __syncthreads();

    // Step 2: Inverse rotation
    float sum = 0.0f;
    for (int j = 0; j < 128; j++) {
        sum += Q_rot[j * 128 + tid] * s_rotated[j];
    }

    // Step 3: Scale and write
    y[tbq_block * 256 + sub_half * 128 + tid] = (dst_t)(sum * norm);
}

// Ensure rotation matrix is initialized (lazy init, thread-safe via CUDA stream ordering)
static void turboq_ensure_init(cudaStream_t stream) {
    if (d_turboq_Q == nullptr) {
        ggml_cuda_turboq_init(stream);
    }
}

template<typename dst_t>
void dequantize_row_tbq3_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    turboq_ensure_init(stream);
    const int64_t nb = k / 256;
    dequantize_block_tbq3_0_kernel<<<nb * 2, 128, 0, stream>>>(vx, y, d_turboq_Q, nb);
}

template<typename dst_t>
void dequantize_row_tbq4_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream) {
    turboq_ensure_init(stream);
    const int64_t nb = k / 256;
    dequantize_block_tbq4_0_kernel<<<nb * 2, 128, 0, stream>>>(vx, y, d_turboq_Q, nb);
}

// Explicit template instantiations
template void dequantize_row_tbq3_0_cuda<float>(const void * vx, float * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq3_0_cuda<half>(const void * vx, half * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq4_0_cuda<float>(const void * vx, float * y, int64_t k, cudaStream_t stream);
template void dequantize_row_tbq4_0_cuda<half>(const void * vx, half * y, int64_t k, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Quantize kernels (f32 -> TBQ for KV cache write path)
//
// Two-pass approach:
//   Pass 1: Compute per-block L2 norms (256 threads per block)
//   Pass 2: Normalize, rotate, quantize, pack (128 threads per sub-block)
// ---------------------------------------------------------------------------

__global__ void turboq_compute_norms_kernel(
    const float * __restrict__ x, float * __restrict__ norms, int64_t num_blocks) {

    const int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    const int tid = threadIdx.x;  // 0..255
    const float * src = x + block_id * 256;

    float val = src[tid];
    float sum_sq = val * val;

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }

    __shared__ float warp_sums[8];  // 256/32 = 8 warps
    if (tid % 32 == 0) {
        warp_sums[tid / 32] = sum_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 8) {
        float s = warp_sums[tid];
        for (int offset = 4; offset > 0; offset >>= 1) {
            s += __shfl_down_sync(0xFF, s, offset);
        }
        if (tid == 0) {
            float norm = sqrtf(s);
            norms[block_id] = (norm < 1e-10f) ? 1e-10f : norm;
        }
    }
}

// TBQ3_0 quantize: normalize, rotate, 3-bit Lloyd-Max, pack
__global__ void quantize_f32_tbq3_0_kernel(
    const float * __restrict__ x, void * __restrict__ vy,
    const float * __restrict__ Q_rot, const float * __restrict__ norms,
    int64_t num_blocks) {

    const int sub_block = blockIdx.x;
    const int tbq_block = sub_block / 2;
    const int sub_half  = sub_block % 2;
    const int tid = threadIdx.x;  // 0..127

    if (tbq_block >= num_blocks) return;

    block_tbq3_0 * y = (block_tbq3_0 *)vy;
    const float * src = x + tbq_block * 256;
    const float norm = norms[tbq_block];
    const float inv_norm = 1.0f / norm;
    const float scale_up = 16.0f;  // sqrt(256)

    // Step 1: Load and normalize
    const int elem = sub_half * 128 + tid;
    float unit_val = src[elem] * inv_norm;

    __shared__ float s_unit[128];
    s_unit[tid] = unit_val;
    __syncthreads();

    // Step 2: Forward rotation: rotated = Q_rot[tid, :] . s_unit[:]
    float rotated = 0.0f;
    for (int j = 0; j < 128; j++) {
        rotated += Q_rot[tid * 128 + j] * s_unit[j];
    }

    // Step 3: Scale and quantize with 3-bit boundaries
    float scaled = rotated * scale_up;
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
    const int qs_offset = sub_half * 48;  // 128*3/8
    if (pos_in_group < 3) {
        // 3 bytes per group, thread 0-2 in each group write one byte
        uint32_t packed = s_packed[group];
        y[tbq_block].qs[qs_offset + group * 3 + pos_in_group] =
            (uint8_t)((packed >> (pos_in_group * 8)) & 0xFF);
    }

    // First sub-block, first thread writes the norm
    if (sub_half == 0 && tid == 0) {
        y[tbq_block].d = __float2half(norm);
    }
}

// TBQ4_0 quantize: normalize, rotate, 4-bit Lloyd-Max, pack nibbles
__global__ void quantize_f32_tbq4_0_kernel(
    const float * __restrict__ x, void * __restrict__ vy,
    const float * __restrict__ Q_rot, const float * __restrict__ norms,
    int64_t num_blocks) {

    const int sub_block = blockIdx.x;
    const int tbq_block = sub_block / 2;
    const int sub_half  = sub_block % 2;
    const int tid = threadIdx.x;

    if (tbq_block >= num_blocks) return;

    block_tbq4_0 * y = (block_tbq4_0 *)vy;
    const float * src = x + tbq_block * 256;
    const float norm = norms[tbq_block];
    const float inv_norm = 1.0f / norm;
    const float scale_up = 16.0f;

    // Step 1: Load and normalize
    const int elem = sub_half * 128 + tid;
    float unit_val = src[elem] * inv_norm;

    __shared__ float s_unit[128];
    s_unit[tid] = unit_val;
    __syncthreads();

    // Step 2: Forward rotation
    float rotated = 0.0f;
    for (int j = 0; j < 128; j++) {
        rotated += Q_rot[tid * 128 + j] * s_unit[j];
    }

    // Step 3: Scale and quantize with 4-bit boundaries
    float scaled = rotated * scale_up;
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
        const int qs_idx = elem / 2;
        y[tbq_block].qs[qs_idx] = s_indices[tid] | (s_indices[tid + 1] << 4);
    }

    // Write norm
    if (sub_half == 0 && tid == 0) {
        y[tbq_block].d = __float2half(norm);
    }
}

// ---------------------------------------------------------------------------
// Copy dispatch functions (called from cpy.cu)
// ---------------------------------------------------------------------------

// Temporary buffer for norms (lazy-allocated, per-stream)
static float * d_turboq_norms = nullptr;
static int64_t d_turboq_norms_size = 0;

static float * turboq_get_norms_buffer(int64_t num_blocks, cudaStream_t stream) {
    if (num_blocks > d_turboq_norms_size) {
        if (d_turboq_norms != nullptr) {
            CUDA_CHECK(cudaFree(d_turboq_norms));
        }
        CUDA_CHECK(cudaMalloc(&d_turboq_norms, num_blocks * sizeof(float)));
        d_turboq_norms_size = num_blocks;
    }
    (void)stream;
    return d_turboq_norms;
}

void ggml_cpy_f32_tbq3_0_cuda(
    const char * cx, char * cdst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    cudaStream_t stream) {

    (void)ne00; (void)ne01; (void)ne02; (void)nb00; (void)nb01; (void)nb02; (void)nb03;
    (void)ne10; (void)ne11; (void)ne12; (void)nb10; (void)nb11; (void)nb12; (void)nb13;

    GGML_ASSERT(ne % 256 == 0);
    turboq_ensure_init(stream);

    const int64_t num_blocks = ne / 256;
    float * norms = turboq_get_norms_buffer(num_blocks, stream);

    // Pass 1: compute norms
    turboq_compute_norms_kernel<<<num_blocks, 256, 0, stream>>>(
        (const float *)cx, norms, num_blocks);

    // Pass 2: quantize
    quantize_f32_tbq3_0_kernel<<<num_blocks * 2, 128, 0, stream>>>(
        (const float *)cx, cdst, d_turboq_Q, norms, num_blocks);
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

    GGML_ASSERT(ne % 256 == 0);
    turboq_ensure_init(stream);

    const int64_t num_blocks = ne / 256;
    float * norms = turboq_get_norms_buffer(num_blocks, stream);

    turboq_compute_norms_kernel<<<num_blocks, 256, 0, stream>>>(
        (const float *)cx, norms, num_blocks);

    quantize_f32_tbq4_0_kernel<<<num_blocks * 2, 128, 0, stream>>>(
        (const float *)cx, cdst, d_turboq_Q, norms, num_blocks);
}

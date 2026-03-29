#pragma once

#include "common.cuh"

// Initialize TurboQuant rotation matrix on device (call once before use)
void ggml_cuda_turboq_init(cudaStream_t stream);

// Free device rotation matrix
void ggml_cuda_turboq_free(void);

// Get device pointer to rotation matrix (128x128, row-major, float32)
const float * ggml_cuda_turboq_get_rotation(void);

// Dequantize dispatchers (bulk, rotation-based — not per-element)
template<typename dst_t>
void dequantize_row_tbq3_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream);

template<typename dst_t>
void dequantize_row_tbq4_0_cuda(const void * vx, dst_t * y, int64_t k, cudaStream_t stream);

// Quantize (f32 -> TBQ) for KV cache write path
void ggml_cpy_f32_tbq3_0_cuda(
    const char * cx, char * cdst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    cudaStream_t stream);

void ggml_cpy_f32_tbq4_0_cuda(
    const char * cx, char * cdst, int64_t ne,
    int64_t ne00, int64_t ne01, int64_t ne02,
    int64_t nb00, int64_t nb01, int64_t nb02, int64_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    int64_t nb10, int64_t nb11, int64_t nb12, int64_t nb13,
    cudaStream_t stream);

// SET_ROWS dispatch for TBQ types (fused, no host-device sync)
template<typename idx_t>
void ggml_set_rows_tbq3_0_cuda(
    const float * src0_d, const idx_t * src1_d, char * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

template<typename idx_t>
void ggml_set_rows_tbq4_0_cuda(
    const float * src0_d, const idx_t * src1_d, char * dst_d,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    size_t nb01, size_t nb02, size_t nb03,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1, size_t nb2, size_t nb3,
    cudaStream_t stream);

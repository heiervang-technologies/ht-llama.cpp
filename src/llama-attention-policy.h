#pragma once

#include "ggml.h"

#include <cstring>

static inline bool llama_gemma4_hybrid_requested(const char * value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Experimental policy: only the short-context, single-stream F16 workload
// measured on Lunar Lake is eligible. Other workloads retain stock attention.
static inline bool llama_gemma4_decompose_attention(
        bool enabled, bool probing, bool flash_attn, bool gemma4,
        ggml_type type_k, ggml_type type_v,
        int64_t n_ctx, int64_t n_seq, int64_t head_dim, int64_t n_query) {
    if (!enabled || probing || !flash_attn || !gemma4 ||
        type_k != GGML_TYPE_F16 || type_v != GGML_TYPE_F16 ||
        n_ctx > 2048 || n_seq != 1 || n_query < 1) {
        return false;
    }
    return (head_dim == 256 || head_dim == 512) &&
           (n_query < 32 || (head_dim == 512 && n_query >= 64));
}

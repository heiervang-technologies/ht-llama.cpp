#include "../src/llama-attention-policy.h"

#include <cstdio>
#include <initializer_list>

int main() {
    GGML_ASSERT(llama_gemma4_hybrid_requested("1"));
    for (const char * value : std::initializer_list<const char *>{nullptr, "", "0", "false", "true", "01", "1x"}) {
        GGML_ASSERT(!llama_gemma4_hybrid_requested(value));
    }
    for (int head : {256, 512}) {
        for (int query : {1, 2, 16, 31, 32, 63, 64, 128, 512}) {
            const bool expected = query < 32 || (head == 512 && query >= 64);
            GGML_ASSERT(llama_gemma4_decompose_attention(true, false, true, true,
                        GGML_TYPE_F16, GGML_TYPE_F16, 2048, 1, head, query) == expected);
            // Probing must retain FA even for the one-token reserve graph.
            GGML_ASSERT(!llama_gemma4_decompose_attention(true, true, true, true,
                        GGML_TYPE_F16, GGML_TYPE_F16, 2048, 1, head, query));
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F32, GGML_TYPE_TBQ4_0}) {
                GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, true, true,
                            GGML_TYPE_F16, type, 2048, 1, head, query));
                GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, true, true,
                            type, GGML_TYPE_F16, 2048, 1, head, query));
            }
            for (int context : {2049, 8192, 16384, 32768}) {
                GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, true, true,
                            GGML_TYPE_F16, GGML_TYPE_F16, context, 1, head, query));
            }
            GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, true, true,
                        GGML_TYPE_F16, GGML_TYPE_F16, 2048, 4, head, query));
        }
    }
    GGML_ASSERT(!llama_gemma4_decompose_attention(false, false, true, true,
                GGML_TYPE_F16, GGML_TYPE_F16, 2048, 1, 512, 1));
    GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, false, true,
                GGML_TYPE_F16, GGML_TYPE_F16, 2048, 1, 512, 1));
    GGML_ASSERT(!llama_gemma4_decompose_attention(true, false, true, false,
                GGML_TYPE_F16, GGML_TYPE_F16, 2048, 1, 512, 1));
    std::puts("Gemma 4 attention boundaries and fallback policy: PASS");
}

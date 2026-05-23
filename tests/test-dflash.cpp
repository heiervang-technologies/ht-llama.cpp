#include "common.h"
#include "log.h"
#include "ggml.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-graph.h"
#include "../src/llama-hparams.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

//
// Test 1: DFlash arch name lookup and not UNKNOWN
//
static void test_dflash_arch_registered() {
    const char * name = llm_arch_name(LLM_ARCH_DFLASH);
    GGML_ASSERT(name != nullptr);
    GGML_ASSERT(std::string(name) == "dflash");
    GGML_ASSERT(LLM_ARCH_DFLASH != LLM_ARCH_UNKNOWN);
    LOG_INF("test_dflash_arch_registered: PASS\n");
}

//
// Test 2: DFlash tensor info lookups
//
static void test_dflash_tensor_infos() {
    {
        const auto & info = llm_tensor_info_for(LLM_TENSOR_DFLASH_FC);
        GGML_ASSERT(info.layer == LLM_TENSOR_LAYER_OUTPUT);
        GGML_ASSERT(info.op == GGML_OP_MUL_MAT);
    }
    {
        const auto & info = llm_tensor_info_for(LLM_TENSOR_DFLASH_HIDDEN_NORM);
        GGML_ASSERT(info.layer == LLM_TENSOR_LAYER_OUTPUT);
        GGML_ASSERT(info.op == GGML_OP_MUL);
    }
    LOG_INF("test_dflash_tensor_infos: PASS\n");
}

//
// Test 3: llama_hparams DFlash defaults
//
static void test_dflash_hparams_defaults() {
    llama_hparams hparams;
    for (int i = 0; i < 16; i++) {
        GGML_ASSERT(hparams.dflash_target_layer_ids[i] == -1);
    }
    GGML_ASSERT(hparams.dflash_block_size == 16);
    GGML_ASSERT(hparams.dflash_mask_token_id == 0);
    LOG_INF("test_dflash_hparams_defaults: PASS\n");
}

//
// Test 4: llama_dflash struct lifecycle
//
static void test_dflash_struct() {
    llama_dflash dflash;
    GGML_ASSERT(dflash.extract_layer_indices.empty());
    GGML_ASSERT(dflash.target_features.empty());
    GGML_ASSERT(dflash.extract_tensors.empty());

    dflash.extract_layer_indices = {0, 4, 8, 12, 16};
    GGML_ASSERT(dflash.extract_layer_indices.size() == 5);
    GGML_ASSERT(dflash.extract_layer_indices[2] == 8);

    dflash.target_features = {0.1f, 0.2f};
    dflash.extract_tensors.resize(2, nullptr);
    dflash.clear();
    GGML_ASSERT(dflash.target_features.empty());
    GGML_ASSERT(dflash.extract_tensors.empty());
    LOG_INF("test_dflash_struct: PASS\n");
}

//
// Test 5: llm_graph_params dflash pointer wiring
//
static void test_dflash_graph_params() {
    llama_dflash dflash;

    // Default-constructed params have null dflash
    {
        llm_graph_params params;
        GGML_ASSERT(params.dflash == nullptr);
    }

    // After setting, pointer is reachable
    dflash.extract_layer_indices = {0, 2};
    {
        llm_graph_params params;
        params.dflash = &dflash;
        GGML_ASSERT(params.dflash != nullptr);
        GGML_ASSERT(params.dflash->extract_layer_indices.size() == 2);
        GGML_ASSERT(params.dflash->extract_layer_indices[0] == 0);
    }

    LOG_INF("test_dflash_graph_params: PASS\n");
}

//
// Test 6: COMMON_SPECULATIVE_TYPE_DFLASH placement
//
static void test_dflash_speculative_type() {
    static_assert(COMMON_SPECULATIVE_TYPE_DFLASH > COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
        "DFlash must be after DRAFT_MTP");
    static_assert(COMMON_SPECULATIVE_TYPE_DFLASH < COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
        "DFlash must be before NGRAM_SIMPLE");
    static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 10,
        "COUNT must be 10 with DFlash");
    LOG_INF("test_dflash_speculative_type: PASS\n");
}

//
// Test 7: llama_context_params target_model defaults
//
static void test_dflash_context_params() {
    llama_context_params params = llama_context_default_params();
    GGML_ASSERT(params.target_model == nullptr);
    params.target_model = reinterpret_cast<const llama_model *>(0x1);
    GGML_ASSERT(params.target_model != nullptr);
    LOG_INF("test_dflash_context_params: PASS\n");
}

//
// Test 8: DFlash API symbols resolve at link time
//
extern "C" {
    LLAMA_API int32_t llama_model_dflash_block_size  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_dflash_mask_token_id(const struct llama_model * model);
}

static void test_dflash_api_symbols() {
    GGML_UNUSED(&llama_model_dflash_block_size);
    GGML_UNUSED(&llama_model_dflash_mask_token_id);
    LOG_INF("test_dflash_api_symbols: PASS\n");
}

int main(void) {
    ggml_time_init();

    test_dflash_arch_registered();
    test_dflash_tensor_infos();
    test_dflash_hparams_defaults();
    test_dflash_struct();
    test_dflash_graph_params();
    test_dflash_speculative_type();
    test_dflash_context_params();
    test_dflash_api_symbols();

    LOG_INF("All DFlash unit tests passed.\n");
    return 0;
}

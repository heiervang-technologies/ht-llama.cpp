#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "../src/llama-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static const std::vector<int> lengths = {1, 2, 16, 31, 32, 63, 64, 128, 1152};

static std::vector<float> logits(llama_context * ctx, const llama_vocab * vocab) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * data = llama_get_logits_ith(ctx, -1);
    GGML_ASSERT(data);
    std::vector<float> result(data, data + n_vocab);
    std::vector<bool> suppressed(n_vocab, false);
    for (llama_token token : vocab->get_suppress_tokens()) {
        if (token >= 0 && token < n_vocab) { suppressed[token] = true; }
    }
    for (int i = 0; i < n_vocab; ++i) {
        if (suppressed[i]) {
            GGML_ASSERT(std::isinf(result[i]) && result[i] < 0);
        } else {
            GGML_ASSERT(std::isfinite(result[i]));
        }
    }
    return result;
}

static void compare(const std::vector<float> & actual, const std::vector<float> & reference, const char * label) {
    GGML_ASSERT(actual.size() == reference.size());
    double error = 0, scale = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isinf(reference[i]) && reference[i] < 0) {
            GGML_ASSERT(actual[i] == reference[i]);
            continue;
        }
        error += std::pow(double(actual[i]) - reference[i], 2);
        scale += double(reference[i])*reference[i];
    }
    const double nmse = error / std::max(scale, 1e-30);
    const bool same_top = std::max_element(actual.begin(), actual.end()) - actual.begin() ==
                          std::max_element(reference.begin(), reference.end()) - reference.begin();
    std::printf("%s: nmse=%.9g top1_equal=%d\n", label, nmse, same_top);
    std::fflush(stdout);
    // Same tolerance as the FLASH_ATTN_EXT backend comparisons. Record top-1
    // separately because nearly tied logits may swap under FP16 arithmetic.
    GGML_ASSERT(nmse < 5e-4);
}

int main(int argc, char ** argv) {
    if (argc != 4 || (std::string(argv[3]) != "cpu" && std::string(argv[3]) != "gpu")) {
        std::fprintf(stderr, "usage: %s MODEL CPU_LOGITS_FILE cpu|gpu\n", argv[0]);
        return 1;
    }
    const bool cpu = std::string(argv[3]) == "cpu";
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = cpu ? 0 : 999;
    auto * model = llama_model_load_from_file(argv[1], mp);
    GGML_ASSERT(model);
    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::string text;
    while (text.size() < 30000) {
        text += "The observatory records the stars each night. Explain how the telescope measures their positions.\n";
    }
    auto tokens = common_tokenize(vocab, text, true, true);
    GGML_ASSERT(tokens.size() > 1200);

    std::vector<std::vector<float>> reference(lengths.size(), std::vector<float>(n_vocab));
    if (!cpu) {
        std::ifstream input(argv[2], std::ios::binary);
        GGML_ASSERT(input);
        for (auto & row : reference) {
            input.read(reinterpret_cast<char *>(row.data()), row.size()*sizeof(float));
            GGML_ASSERT(input);
        }
        GGML_ASSERT(input.peek() == std::char_traits<char>::eof());
    }
    struct configuration { const char * name; const char * gate; llama_flash_attn_type fa; ggml_type cache; };
    const std::vector<configuration> configs = cpu ? std::vector<configuration>{
        {"cpu", "0", LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16},
    } : std::vector<configuration>{
        {"off", "0", LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16},
        {"on", "0", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_F16},
        {"hybrid", "1", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_F16},
        {"hybrid-auto", "1", LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
        {"hybrid-q8", "1", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_Q8_0},
        {"hybrid-q4", "1", LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_Q4_0},
    };
    for (const auto & config : configs) {
#ifdef _WIN32
        _putenv_s("LLAMA_VK_GEMMA4_HYBRID_FA", config.gate);
#else
        setenv("LLAMA_VK_GEMMA4_HYBRID_FA", config.gate, 1);
#endif
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = 2048;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.op_offload = !cpu;
        cp.offload_kqv = !cpu;
        cp.flash_attn_type = config.fa;
        cp.type_k = cp.type_v = config.cache;
        auto * ctx = llama_init_from_model(model, cp);
        GGML_ASSERT(ctx);
        for (size_t i = 0; i < lengths.size(); ++i) {
            llama_memory_clear(llama_get_memory(ctx), true);
            GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data(), lengths[i])) == 0);
            auto result = logits(ctx, vocab);
            if (cpu) {
                reference[i] = result;
            } else if (config.cache == GGML_TYPE_F16) {
                compare(result, reference[i], (std::string(config.name) + " n=" + std::to_string(lengths[i])).c_str());
            }
            // Dirty then free/reuse cells after SWA has wrapped, simulating a
            // rejected speculative suffix while keeping the prefix resident.
            if (lengths[i] > 1024) {
                auto memory = llama_get_memory(ctx);
                const int start = 1100;
                GGML_ASSERT(llama_memory_seq_rm(memory, 0, start, -1));
                GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data() + 20, lengths[i] - start)) == 0);
                GGML_ASSERT(llama_memory_seq_rm(memory, 0, start, -1));
                GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data() + start, lengths[i] - start)) == 0);
                compare(logits(ctx, vocab), result, (std::string(config.name) + " reused KV").c_str());
            }
        }
        llama_free(ctx);
    }
    if (cpu) {
        std::ofstream output(argv[2], std::ios::binary);
        for (const auto & row : reference) {
            output.write(reinterpret_cast<const char *>(row.data()), row.size()*sizeof(float));
        }
        GGML_ASSERT(output);
    }
    llama_model_free(model);
    llama_backend_free();
    std::puts("Gemma 4 device validation: PASS");
}

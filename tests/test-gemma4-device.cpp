#include "common.h"
#include "chat.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static const std::vector<int> lengths = {1, 2, 16, 31, 32, 63, 64, 128, 1152};

static std::vector<bool> suppression_mask(const char * path, int n_vocab) {
    std::vector<bool> suppressed(n_vocab, false);
    auto * metadata = gguf_init_from_file(path, {true, nullptr});
    GGML_ASSERT(metadata);
    const int64_t key = gguf_find_key(metadata, "tokenizer.ggml.suppress_tokens");
    if (key >= 0) {
        GGML_ASSERT(gguf_get_arr_type(metadata, key) == GGUF_TYPE_INT32);
        const auto * tokens = static_cast<const int32_t *>(gguf_get_arr_data(metadata, key));
        for (size_t i = 0; i < gguf_get_arr_n(metadata, key); ++i) {
            if (tokens[i] >= 0 && tokens[i] < n_vocab) { suppressed[tokens[i]] = true; }
        }
    }
    gguf_free(metadata);
    return suppressed;
}

static std::vector<float> logits(llama_context * ctx, const std::vector<bool> & suppressed) {
    const float * data = llama_get_logits_ith(ctx, -1);
    GGML_ASSERT(data);
    std::vector<float> result(data, data + suppressed.size());
    for (size_t i = 0; i < suppressed.size(); ++i) {
        if (suppressed[i]) {
            GGML_ASSERT(std::isinf(result[i]) && result[i] < 0);
        } else {
            GGML_ASSERT(std::isfinite(result[i]));
        }
    }
    return result;
}

static bool compare(const std::vector<float> & actual, const std::vector<float> & reference, const char * label) {
    GGML_ASSERT(actual.size() == reference.size());
    double error = 0, scale = 0;
    const double max_actual = *std::max_element(actual.begin(), actual.end());
    const double max_reference = *std::max_element(reference.begin(), reference.end());
    double sum_actual = 0, sum_reference = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        sum_actual += std::exp(actual[i] - max_actual);
        sum_reference += std::exp(reference[i] - max_reference);
    }
    const double log_z_actual = max_actual + std::log(sum_actual);
    const double log_z_reference = max_reference + std::log(sum_reference);
    double kl = 0, tv = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isinf(reference[i]) && reference[i] < 0) {
            GGML_ASSERT(actual[i] == reference[i]);
            continue;
        }
        const double log_p = reference[i] - log_z_reference;
        const double log_q = actual[i] - log_z_actual;
        const double p = std::exp(log_p), q = std::exp(log_q);
        kl += p * (log_p - log_q);
        tv += std::abs(p - q) / 2;
        error += std::pow(double(actual[i]) - reference[i], 2);
        scale += double(reference[i])*reference[i];
    }
    const double nmse = error / std::max(scale, 1e-30);
    const bool same_top = std::max_element(actual.begin(), actual.end()) - actual.begin() ==
                          std::max_element(reference.begin(), reference.end()) - reference.begin();
    std::printf("%s: nmse=%.9g kl=%.9g tv=%.9g top1_equal=%d\n", label, nmse, kl, tv, same_top);
    std::fflush(stdout);
    // Same tolerance as the FLASH_ATTN_EXT backend comparisons. Record top-1
    // separately because nearly tied logits may swap under FP16 arithmetic.
    return nmse < 5e-4;
}

int main(int argc, char ** argv) {
    if ((argc != 4 && argc != 5) || (std::string(argv[3]) != "cpu" && std::string(argv[3]) != "gpu")) {
        std::fprintf(stderr, "usage: %s MODEL CPU_LOGITS_FILE cpu|gpu [CONFIG]\n", argv[0]);
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
    const auto suppressed = suppression_mask(argv[1], n_vocab);
    auto templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs chat;
    common_chat_msg message;
    message.role = "user";
    message.content = "Write an observing log with numbered entries. Explain calibration, weather, star positions, and uncertainty.";
    chat.messages.push_back(message);
    chat.enable_thinking = false;
    chat.chat_template_kwargs["enable_thinking"] = "false";
    const auto formatted = common_chat_templates_apply(templates.get(), chat);
    auto prefix = common_tokenize(vocab, formatted.prompt, true, true);
    std::string text = "The observing log separates measurements from interpretation. Each entry records the exposure and the checks used to assess its reliability.\n\n";
    for (int i = 1; i <= 128; ++i) {
        text += "Entry " + std::to_string(i) + ": At minute " + std::to_string(i * 3) +
                ", the telescope recorded " + std::to_string(1000 + i * 17) +
                " counts. The observer checked tracking, background light, and calibration before comparing this frame with the reference exposure.\n";
    }
    auto tokens = common_tokenize(vocab, text, false, true);
    GGML_ASSERT(tokens.size() > 1200 && prefix.size() + lengths.back() < 2048);

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
    bool passed = true;
    auto gpu_reference = reference;
    bool have_gpu_reference = false;
    int ran = 0;
    for (const auto & config : configs) {
        if (argc == 5 && std::string(argv[4]) != config.name) { continue; }
        ++ran;
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
        std::ofstream dump;
        if (!cpu) {
            dump.open(std::string(argv[2]) + "." + config.name, std::ios::binary);
            GGML_ASSERT(dump);
        }
        for (size_t i = 0; i < lengths.size(); ++i) {
            llama_memory_clear(llama_get_memory(ctx), true);
            GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(prefix.data(), prefix.size())) == 0);
            GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data(), lengths[i])) == 0);
            auto result = logits(ctx, suppressed);
            if (!cpu) {
                dump.write(reinterpret_cast<const char *>(result.data()), result.size()*sizeof(float));
                GGML_ASSERT(dump);
            }
            if (cpu) {
                reference[i] = result;
                std::printf("cpu n=%d: reference recorded\n", lengths[i]);
                std::fflush(stdout);
            } else if (config.cache == GGML_TYPE_F16) {
                passed &= compare(result, reference[i], (std::string(config.name) + " n=" + std::to_string(lengths[i])).c_str());
            }
            if (!cpu && config.cache == GGML_TYPE_F16) {
                if (std::string(config.name) == "off") {
                    gpu_reference[i] = result;
                    have_gpu_reference = true;
                } else if (have_gpu_reference) {
                    passed &= compare(result, gpu_reference[i],
                        (std::string(config.name) + " vs GPU-off n=" + std::to_string(lengths[i])).c_str());
                }
            }
            // Dirty then free/reuse cells after SWA has wrapped, simulating a
            // rejected speculative suffix while keeping the prefix resident.
            if (lengths[i] > 1024) {
                auto memory = llama_get_memory(ctx);
                const int start = 1100;
                GGML_ASSERT(llama_memory_seq_rm(memory, 0, prefix.size() + start, -1));
                GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data() + 20, lengths[i] - start)) == 0);
                GGML_ASSERT(llama_memory_seq_rm(memory, 0, prefix.size() + start, -1));
                GGML_ASSERT(llama_decode(ctx, llama_batch_get_one(tokens.data() + start, lengths[i] - start)) == 0);
                passed &= compare(logits(ctx, suppressed), result, (std::string(config.name) + " reused KV").c_str());
            }
        }
        llama_free(ctx);
    }
    GGML_ASSERT(ran > 0);
    if (cpu) {
        std::ofstream output(argv[2], std::ios::binary);
        for (const auto & row : reference) {
            output.write(reinterpret_cast<const char *>(row.data()), row.size()*sizeof(float));
        }
        GGML_ASSERT(output);
    }
    llama_model_free(model);
    llama_backend_free();
    std::puts(passed ? "Gemma 4 device validation: PASS" : "Gemma 4 device validation: FAIL");
    return passed ? 0 : 1;
}

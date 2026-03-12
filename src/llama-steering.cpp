#include "llama-steering.h"
#include "llama-impl.h"

#include <vector>
#include <string>

int32_t llama_steering_hint_inject(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_pos              inject_pos,
        const llama_token    * tokens,
        int32_t                n_tokens) {
    if (!ctx || !tokens || n_tokens <= 0) {
        return -1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        return -2;
    }

    // If inject_pos is -1, inject after the current max position
    if (inject_pos < 0) {
        inject_pos = llama_memory_seq_pos_max(mem, seq_id) + 1;
    }

    // Step 1: Shift existing KV cache entries to make room
    // All positions >= inject_pos get shifted by n_tokens
    // p1 = -1 means [inject_pos, inf)
    llama_memory_seq_add(mem, seq_id, inject_pos, -1, n_tokens);

    // Step 2: Build a batch with hint tokens at the gap positions
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = inject_pos + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i]    = 0;  // no logits needed for hint tokens
    }
    batch.n_tokens = n_tokens;

    // Step 3: Decode the hint tokens to fill the KV cache gap
    int32_t ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to decode steering hint tokens (ret=%d), rolling back position shift\n", __func__, ret);
        // Undo the position shift to restore KV cache consistency
        llama_memory_seq_add(mem, seq_id, inject_pos + n_tokens, -1, -n_tokens);
        return -3;
    }

    return 0;
}

int32_t llama_steering_hint_prepare(
        const struct llama_model * model,
        const char               * chat_template,
        const char               * role,
        const char               * text,
        llama_token              * out_tokens,
        int32_t                    max_tokens) {
    if (!model || !role || !text) {
        return -1;
    }

    // Build a single-message chat and apply template
    llama_chat_message msg;
    msg.role    = role;
    msg.content = text;

    // First call to get required buffer size
    int32_t tmpl_len = llama_chat_apply_template(
        chat_template, &msg, 1, false, nullptr, 0);

    if (tmpl_len < 0) {
        LLAMA_LOG_ERROR("%s: failed to apply chat template\n", __func__);
        return -2;
    }

    // Apply template to get formatted string
    std::vector<char> buf(tmpl_len + 1);
    llama_chat_apply_template(
        chat_template, &msg, 1, false, buf.data(), buf.size());

    std::string formatted(buf.data(), tmpl_len);

    // Tokenize with special token parsing enabled
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // First get token count
    int32_t n = llama_tokenize(vocab, formatted.c_str(), formatted.length(),
                               nullptr, 0, false, true);

    if (n < 0) {
        n = -n;  // llama_tokenize returns negative count when buffer too small
    }

    if (!out_tokens) {
        return n;  // caller just wants the count
    }

    if (n > max_tokens) {
        return -3;  // buffer too small
    }

    // Actually tokenize
    int32_t actual = llama_tokenize(vocab, formatted.c_str(), formatted.length(),
                                    out_tokens, max_tokens, false, true);

    if (actual < 0) {
        LLAMA_LOG_ERROR("%s: tokenization failed\n", __func__);
        return -4;
    }

    return actual;
}

#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Inject steering hint tokens into active generation at a given context position.
// This shifts existing KV cache entries at positions >= inject_pos by n_tokens,
// then decodes the hint tokens into the gap.
//
// Parameters:
//   ctx        - the llama context with active generation
//   seq_id     - sequence to inject into
//   inject_pos - position to inject at (-1 = after current max position)
//   tokens     - array of pre-tokenized hint tokens (must include chat template wrapping)
//   n_tokens   - number of tokens
//
// Returns 0 on success, negative on error.
LLAMA_API int32_t llama_steering_hint_inject(
    struct llama_context * ctx,
    llama_seq_id           seq_id,
    llama_pos              inject_pos,
    const llama_token    * tokens,
    int32_t                n_tokens);

// Convenience: wrap text with chat template and tokenize for use with llama_steering_hint_inject.
// Writes tokenized output to out_tokens. Returns number of tokens written, or negative on error.
// If out_tokens is NULL, returns the number of tokens needed.
LLAMA_API int32_t llama_steering_hint_prepare(
    const struct llama_model * model,
    const char               * chat_template,  // NULL = use model default
    const char               * role,           // "user" or "system"
    const char               * text,
    llama_token              * out_tokens,
    int32_t                    max_tokens);

#ifdef __cplusplus
}
#endif

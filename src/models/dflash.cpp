#include "models.h"

// ==========================================================================
// DFlash encoder graph — fuses extracted target features into draft token embedding
// ==========================================================================

ggml_tensor * llm_build_dflash_encode::build_inp_embd() const {
    int64_t n_target_layer_ids = 0;
    for (int i = 0; i < 16; i++) {
        if (hparams.dflash_target_layer_ids[i] != -1) n_target_layer_ids++;
        else break;
    }
    const int64_t n_embd_target_features = n_target_layer_ids * n_embd;

    auto inp_target = std::make_unique<llm_graph_input_embd>(n_embd_target_features);
    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_target_features, n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);
    res->add_input(std::move(inp_target));
    return cur;
}

llm_build_dflash_encode::llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd();
    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);
    cur = build_norm(cur, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "hidden_norm_out", -1);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// ==========================================================================
// DFlash decoder graph — cross-attention draft token generation
// ==========================================================================

llm_build_dflash_decode::llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // env-gated experiment: re-apply layer.attn_norm to fused_target before each layer's wk/wv
    // hypothesis: drafter trained with per-layer ctx renorm; without it K_ctx/V_ctx are wrong-scaled past layer 0
    static const bool dflash_per_layer_renorm = []() {
        const char * e = std::getenv("LLAMA_DFLASH_PER_LAYER_RENORM");
        return e && e[0] != '\0' && e[0] != '0';
    }();

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = model.hparams.dflash_n_target_features != 0 ? model.hparams.dflash_n_target_features : n_embd * llama_model_dflash_n_target_layers(&model);
    const int64_t ctx_len = (cross && !cross->v_embd.empty()) ? cross->n_enc : n_ctx;
    const int64_t n_kv_total = ctx_len + n_tokens;

    const bool dflash_has_swa = hparams.is_swa_any() && hparams.n_swa > 0;
    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(
        cross, ctx_len, n_tokens, dflash_has_swa ? (int64_t)hparams.n_swa : 0);
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16) : inp_dflash->kq_mask;

    if (dflash_has_swa) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        inp_dflash->kq_mask_swa_cnv = cparams.flash_attn ? ggml_cast(ctx0, inp_dflash->kq_mask_swa, GGML_TYPE_F16) : inp_dflash->kq_mask_swa;
    }

    ggml_tensor * kq_mask       = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa   = inp_dflash->kq_mask_swa_cnv;
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    GGML_ASSERT(model.tok_embd != nullptr && "DFlash decoder requires target model's tok_embd");
    // build_inp_embd auto-applies hparams.f_embedding_scale when non-zero — this is how
    // we inherit Gemma4's sqrt(n_embd) noise embedding normalization via the cross-binding
    // in llama-context.cpp. See vLLM PR #41703 for the architectural rationale.
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_noise_embd", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * fused_target = build_lora_mm(model.fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(fused_target, "fused_target", -1);

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * ctx_src = fused_target;
        if (dflash_per_layer_renorm) {
            ctx_src = build_norm(fused_target, layer.attn_norm, NULL, LLM_NORM_RMS, il);
            cb(ctx_src, "ctx_attn_norm", il);
        }

        // Q from noise only
        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
        if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);

        ggml_tensor * K_noise = build_lora_mm(layer.wk, cur);
        if (layer.wk_b) {
            K_noise = ggml_add(ctx0, K_noise, layer.wk_b);
        }
        K_noise = ggml_reshape_3d(ctx0, K_noise, n_embd_head, n_head_kv, n_tokens);
        K_noise = build_norm(K_noise, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        K_noise = ggml_rope_ext(
                ctx0, K_noise, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(K_noise, "Kcur_noise", il);

        ggml_tensor * K_ctx = build_lora_mm(layer.wk, ctx_src);
        if (layer.wk_b) { K_ctx = ggml_add(ctx0, K_ctx, layer.wk_b); }
        K_ctx = ggml_reshape_3d(ctx0, K_ctx, n_embd_head, n_head_kv, ctx_len);
        K_ctx = build_norm(K_ctx, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        K_ctx = ggml_rope_ext(
                ctx0, K_ctx, pos_ctx, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(K_ctx, "Kcur_ctx", il);

        ggml_tensor * V_noise = build_lora_mm(layer.wv, cur);
        if (layer.wv_b) {
            V_noise = ggml_add(ctx0, V_noise, layer.wv_b);
        }
        V_noise = ggml_reshape_3d(ctx0, V_noise, n_embd_head, n_head_kv, n_tokens);
        cb(V_noise, "Vcur_noise", il);

        ggml_tensor * V_ctx = build_lora_mm(layer.wv, ctx_src);
        if (layer.wv_b) { V_ctx = ggml_add(ctx0, V_ctx, layer.wv_b); }
        V_ctx = ggml_reshape_3d(ctx0, V_ctx, n_embd_head, n_head_kv, ctx_len);
        cb(V_ctx, "Vcur_ctx", il);

        ggml_tensor * Kcur = ggml_concat(ctx0, K_ctx, K_noise, 2);
        ggml_tensor * Vcur = ggml_concat(ctx0, V_ctx, V_noise, 2);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        // Per-layer mask selection: SWA layers use the windowed mask (kq_mask_swa) so noise
        // tokens only attend to ctx tokens within the trained sliding window.  Dense layers
        // use the full mask.  At ctx_len <= n_swa this is a no-op (no key is windowed out).
        ggml_tensor * layer_kq_mask = (kq_mask_swa && hparams.is_swa(il)) ? kq_mask_swa : kq_mask;
        cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, layer_kq_mask, nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        if (layer.wo_b) { cur = ggml_add(ctx0, cur, layer.wo_b); }
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_inp = cur;
        cur = build_norm(cur, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    if (model.output) {
        cur = build_lora_mm(model.output, cur, model.output_s);

        // Apply target-architecture-specific final logit softcap (Gemma4: 30.0).
        // Monotonic transform, so does not change argmax for greedy sampling, but
        // matches the distribution the drafter was trained against. See vLLM PR #41703.
        if (hparams.f_final_logit_softcapping != 0.0f) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
            cur = ggml_tanh(ctx0, cur);
            cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
        }

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

// ==========================================================================
// DFlash model class — load_arch_hparams, load_arch_tensors, build_arch_graph
// ==========================================================================

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false);
    ml.get_key(LLM_KV_DFLASH_N_TARGET_FEATURES, hparams.dflash_n_target_features, false);
    if (!ml.get_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, hparams.dflash_target_layer_ids, true)) {
        throw std::runtime_error("missing DFlash target_layer_ids");
    }

    // Sliding Window Attention: drafter GGUF may carry per-layer sliding_window_pattern
    // (e.g. Anbeeld Gemma4 drafter: [T,T,T,T,F] with sliding_window=2048).  Latent at small
    // ctx_window (<=2048) but required for correctness once ctx grows past the window.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        // Re-use std::array<int,16> (already template-instantiated by dflash_target_layer_ids)
        // to receive the BOOL/INT array.  Filled with 0 by default so unset slots are dense.
        std::array<int, 16> pattern{};
        if (ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern, false)) {
            const uint32_t n = std::min<uint32_t>(pattern.size(), hparams.n_layer());
            for (uint32_t il = 0; il < n; ++il) {
                hparams.is_swa_impl[il] = pattern[il] != 0 ? 1u : 0u;
            }
        } else {
            // No per-layer pattern: assume all layers are SWA.
            for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
                hparams.is_swa_impl[il] = 1u;
            }
        }
    }
}

void llama_model_dflash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    int n_target_layer_ids = 0;
    for (int i = 0; i < 16; i++) {
        if (hparams.dflash_target_layer_ids[i] != -1) n_target_layer_ids++;
        else break;
    }

    const int64_t n_embd_target_features = hparams.dflash_n_target_features != 0 ? hparams.dflash_n_target_features : n_embd * n_target_layer_ids;

    // Feature fusion layer: concatenated target features -> hidden state
    fc = create_tensor(tn(LLM_TENSOR_DFLASH_FC, "weight"), {n_embd_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    // Token embeddings and output — shared with target model at runtime
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,       "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,  "weight"), {n_embd}, 0);

    // Layers for draft generation
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        auto & layer = layers[il];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", il), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", il), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", il), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), {n_embd_head_k}, TENSOR_NOT_REQUIRED);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), {n_embd_head_k}, TENSOR_NOT_REQUIRED);

        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, (int64_t)hparams.n_ff(il)}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, (int64_t)hparams.n_ff(il)}, TENSOR_NOT_REQUIRED);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {(int64_t)hparams.n_ff(il), n_embd}, TENSOR_NOT_REQUIRED);
    }

    ml.done_getting_tensors();
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    if (params.dflash && params.cross && !params.cross->v_embd.empty()) {
        return std::make_unique<llm_build_dflash_decode>(*this, params);
    }
    return std::make_unique<llm_build_dflash_encode>(*this, params);
}

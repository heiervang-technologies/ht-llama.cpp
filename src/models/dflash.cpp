#include "models.h"

// ==========================================================================
// DFlash encoder graph — fuses extracted target features into draft token embedding
// ==========================================================================

ggml_tensor * llm_build_dflash_encode::build_inp_embd() const {
    const int64_t n_target_layer_ids = (int64_t) hparams.dflash_target_layer_ids.size();
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
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Noise tokens [MASK]
    GGML_ASSERT(model.target_tok_embd != nullptr && "DFlash decoder requires target model's tok_embd");
    ggml_tensor * noise_embd = build_inp_embd(model.target_tok_embd);
    cb(noise_embd, "inp_noise_embd", -1);

    // Target context via llama_cross (filled from accumulated_target_ctx)
    ggml_tensor * target_ctx = build_inp_cross_embd();
    const int64_t n_ctx = target_ctx->ne[1];

    ggml_tensor * inpL = noise_embd;
    const int64_t n_tokens_kv = n_ctx + n_tokens;

    // Position tensor covering target_ctx + noise
    ggml_tensor * inp_pos_full = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens_kv);
    ggml_set_input(inp_pos_full);
    cb(inp_pos_full, "inp_pos_full", -1);

    // Q positions: last n_tokens entries (noise only)
    ggml_tensor * inp_pos_q = ggml_view_1d(ctx0, inp_pos_full, n_tokens,
            n_ctx * ggml_element_size(inp_pos_full));

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        // Q from noise only
        ggml_tensor * Qcur = build_lora_mm(layer.wq, noise_norm);
        if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
        cb(Qcur, "Qcur", il);

        // K = concat(k_proj(target_ctx), k_proj(noise))
        ggml_tensor * K_tgt   = build_lora_mm(layer.wk, target_ctx);
        ggml_tensor * K_noise = build_lora_mm(layer.wk, noise_norm);
        if (layer.wk_b) {
            K_tgt   = ggml_add(ctx0, K_tgt,   layer.wk_b);
            K_noise = ggml_add(ctx0, K_noise, layer.wk_b);
        }
        ggml_tensor * Kcur = ggml_concat(ctx0, K_tgt, K_noise, 1);
        cb(Kcur, "Kcur", il);

        // V = concat(v_proj(target_ctx), v_proj(noise))
        ggml_tensor * V_tgt   = build_lora_mm(layer.wv, target_ctx);
        ggml_tensor * V_noise = build_lora_mm(layer.wv, noise_norm);
        if (layer.wv_b) {
            V_tgt   = ggml_add(ctx0, V_tgt,   layer.wv_b);
            V_noise = ggml_add(ctx0, V_noise, layer.wv_b);
        }
        ggml_tensor * Vcur = ggml_concat(ctx0, V_tgt, V_noise, 1);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_kv);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        cb(Kcur, "Kcur_normed", il);

        // RoPE: K uses full positions, Q uses last n_tokens
        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos_full, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Kcur, "Kcur_rope", il);

        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos_q, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur_rope", il);

        // Full attention (no causal mask)
        ggml_build_forward_expand(gf, Qcur);
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        ggml_tensor * cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "kqv_out", il);

        cur = build_lora_mm(layer.wo, cur);
        if (layer.wo_b) { cur = ggml_add(ctx0, cur, layer.wo_b); }
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_res", il);

        ggml_tensor * ffn_inp = cur;
        cur = build_norm(cur, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

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

    if (model.target_output) {
        cur = build_lora_mm(model.target_output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
}

// ==========================================================================
// DFlash model class — load_arch_hparams, load_arch_tensors, build_arch_graph
// ==========================================================================

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false);
    if (!ml.get_key_or_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, hparams.dflash_target_layer_ids, 5, true)) {
        throw std::runtime_error("missing DFlash target_layer_ids");
    }
}

void llama_model_dflash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_target_features = n_embd * (int64_t)hparams.dflash_target_layer_ids.size();

    // Feature fusion layer: concatenated target features -> hidden state
    fc = create_tensor(tn(LLM_TENSOR_DFLASH_FC, "weight"), {n_embd_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    // Token embeddings and output — shared with target model at runtime
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,       "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,  "weight"), {n_embd}, 0);

    // Single attention layer for draft generation
    auto & layer = layers[0];
    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", 0), {n_embd}, 0);
    create_tensor_qkv(layer, 0, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", 0), {n_embd_head_k * n_head, n_embd}, 0);

    ml.done_getting_tensors();
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    if (params.dflash && params.cross && !params.cross->v_embd.empty()) {
        return std::make_unique<llm_build_dflash_decode>(*this, params);
    }
    return std::make_unique<llm_build_dflash_encode>(*this, params);
}

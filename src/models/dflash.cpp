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
    cb(cur, 'inp_embd', -1);
    res->add_input(std::move(inp_target));
    return cur;
}

llm_build_dflash_encode::llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd();
    cur = build_lora_mm(model.fc, cur);
    cb(cur, 'fc_out', -1);
    cur = build_norm(cur, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, 'hidden_norm_out', -1);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// ==========================================================================
// DFlash decoder graph — cross-attention draft token generation
// ==========================================================================

llm_build_dflash_decode::llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Noise embeddings as input
    {
        auto inp_ffn = std::make_unique<llm_graph_input_embd>(n_embd);
        inp_ffn->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inp_ffn->embd, 'inp_embd');
        ggml_set_input(inp_ffn->embd);
        cur = inp_ffn->embd;
        res->add_input(std::move(inp_ffn));
    }
    inpL = cur;

    // Positional encoding for full sequence (target context tokens + noise tokens)
    auto inp_pos = std::make_unique<llm_graph_input_pos>();
    inp_pos->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hparams.n_ctx_train);
    ggml_set_name(inp_pos->pos, 'inp_pos_full');
    ggml_set_input(inp_pos->pos);
    res->add_input(std::move(inp_pos));

    // Cross-attn target context
    cur = build_inp_cross_embd();
    cb(cur, 'inp_cross_embd', -1);

    cur = inpL;

    const auto & layer = model.layers[0];

    cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, 0);
    cb(cur, 'attn_norm', 0);

    ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
    cb(Qcur, 'Qcur', 0);

    if (layer.wq_b) { Qcur = ggml_add(ctx0, Qcur, layer.wq_b); }
    cb(Qcur, 'Qcur_b', 0);

    ggml_tensor * K_tgt = build_lora_mm(layer.wk, cross->v_embd, cross->n_embd, cross->n_enc);
    ggml_tensor * K_noise = build_lora_mm(layer.wk, inpL);
    if (layer.wk_b) {
        K_tgt   = ggml_add(ctx0, K_tgt,   layer.wk_b);
        K_noise = ggml_add(ctx0, K_noise, layer.wk_b);
    }
    ggml_tensor * Kcur = ggml_concat(ctx0, K_tgt, K_noise, 1);
    cb(Kcur, 'Kcur', 0);

    ggml_tensor * V_tgt = build_lora_mm(layer.wv, cross->v_embd, cross->n_embd, cross->n_enc);
    ggml_tensor * V_noise = build_lora_mm(layer.wv, inpL);
    if (layer.wv_b) {
        V_tgt   = ggml_add(ctx0, V_tgt,   layer.wv_b);
        V_noise = ggml_add(ctx0, V_noise, layer.wv_b);
    }
    ggml_tensor * Vcur = ggml_concat(ctx0, V_tgt, V_noise, 1);
    cb(Vcur, 'Vcur', 0);

    // Position encoding on target + noise positions
    ggml_tensor * pos_tgt   = ggml_arange(ctx0, 0, (float)cross->n_enc, 1, GGML_TYPE_I32);
    ggml_tensor * pos_noise = ggml_arange(ctx0, (float)cross->n_enc, (float)(cross->n_enc + n_tokens), 1, GGML_TYPE_I32);
    ggml_tensor * pos_full = ggml_concat(ctx0, pos_tgt, pos_noise, 0);

    Kcur = ggml_get_rows(ctx0, Kcur, pos_full);
    Vcur = ggml_get_rows(ctx0, Vcur, pos_full);

    cur = build_attn(inp_pos->pos, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     nullptr, nullptr, 0, n_embd_head, n_embd_head, 1, nullptr, nullptr, nullptr, 0);
    cb(cur, 'kqv_out', 0);

    cur = build_lora_mm(layer.wo, cur);
    if (layer.wo_b) { cur = ggml_add(ctx0, cur, layer.wo_b); }
    cur = ggml_add(ctx0, cur, inpL);
    cb(cur, 'attn_res', 0);

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, 'result_norm', -1);

    cur = build_lora_mm(model.output, cur);
    cb(cur, 'result_output', -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}

// ==========================================================================
// DFlash model class — load_arch_hparams, load_arch_tensors, build_arch_graph
// ==========================================================================

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false);
    if (!ml.get_key_or_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, hparams.dflash_target_layer_ids, 5, true)) {
        throw std::runtime_error('missing DFlash target_layer_ids');
    }
}

void llama_model_dflash::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_target_features = hparams.n_embd * (int64_t)hparams.dflash_target_layer_ids.size();

    // Feature fusion layer: concatenated target features -> hidden state
    fc = create_tensor(tn(LLM_TENSOR_DFLASH_FC, 'weight'), {n_embd_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, 'weight'), {n_embd}, 0);

    // Token embeddings and output — shared with target model at runtime
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  'weight'), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,       'weight'), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,  'weight'), {n_embd}, 0);

    // Single attention layer for draft generation
    auto & layer = layers[0];
    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, 'weight', 0), {n_embd}, 0);
    create_tensor_qkv(layer, 0, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, 'weight', 0), {n_embd_head_k * n_head, n_embd}, 0);

    ml.done_getting_tensors();
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    if (params.dflash && params.cross && !params.cross->v_embd.empty()) {
        return std::make_unique<llm_build_dflash_decode>(*this, params);
    }
    return std::make_unique<llm_build_dflash_encode>(*this, params);
}

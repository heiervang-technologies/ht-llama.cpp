#include "models.h"

// === DFlash model class (post-refactor pattern) ===

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false);
    if (!ml.get_key_or_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, hparams.dflash_target_layer_ids, 5, true)) {
        throw std::runtime_error("missing DFlash target_layer_ids");
    }

    LLAMA_LOG_INFO("%s: DFlash block_size=%u mask_token_id=%u target_layers=[%d,%d,%d,%d,%d]\n",
        __func__, hparams.dflash_block_size, hparams.dflash_mask_token_id,
        hparams.dflash_target_layer_ids[0], hparams.dflash_target_layer_ids[1],
        hparams.dflash_target_layer_ids[2], hparams.dflash_target_layer_ids[3],
        hparams.dflash_target_layer_ids[4]);
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // Feature fusion: target_features (n_target_layer_ids * n_embd) -> n_embd
    const int64_t n_embd_target_features = hparams.n_embd * (int64_t)hparams.dflash_target_layer_ids.size();
    fc = create_tensor(tn(LLM_TENSOR_DFLASH_FC, "weight"), {n_embd_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    // Token embedding layer (shared with target model at runtime)
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // Output layer (shared with target model at runtime)
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Single attention layer for draft token generation
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// === DFlash encoder graph ===

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

// === DFlash decoder graph (single layer self-attn + cross-attn draft) ===

llama_model_dflash::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_embd_head_k = hparams.n_embd_head_k();
    const int64_t n_rot_grouped = n_rot / n_embd_head_k;
    const int64_t n_head_kv = hparams.n_head_kv();

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // Input embeddings (combined noise + g_embd from encoder)
    {
        auto inp_ffn = std::make_unique<llm_graph_input_embd>(n_embd);
        inp_ffn->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inp_ffn->embd, "inp_embd");
        ggml_set_input(inp_ffn->embd);
        cur = inp_ffn->embd;
        res->add_input(std::move(inp_ffn));
    }
    inpL = cur;

    // Input position for decoder (used in ROPE)
    auto inp_pos = std::make_unique<llm_graph_input_pos>();
    inp_pos->pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hparams.n_ctx_train);
    ggml_set_name(inp_pos->pos, "inp_pos_full");
    ggml_set_input(inp_pos->pos);
    cur = inp_pos->pos;
    res->add_input(std::move(inp_pos));

    // Cross-attention: target context features
    cur = build_inp_cross_embd();
    cb(cur, "inp_cross_embd", -1);

    cur = inpL;

    // Self-attention draft layers
    const auto & layer = model.layers[0];

    cur = build_norm(cur, layer.attn_norm, NULL, LLM_NORM_RMS, 0);
    cb(cur, "attn_norm", 0);

    // Q from noise only
    ggml_tensor * Qcur = build_lora_mm(layer.wq, cur);
    cb(Qcur, "Qcur", 0);

    // K/V from cross-attention (target context + noise)
    ggml_tensor * Kcur = build_lora_mm(layer.wk, cross->v_embd, cross->n_embd, cross->n_enc + n_tokens);
    cb(Kcur, "Kcur", 0);

    ggml_tensor * Vcur = build_lora_mm(layer.wv, cross->v_embd, cross->n_embd, cross->n_enc + n_tokens);
    cb(Vcur, "Vcur", 0);

    cur = build_attn(inp_pos->pos, nullptr, nullptr, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                     nullptr, nullptr, 0, n_embd_head, n_embd_head_k, n_rot_grouped, nullptr, nullptr, nullptr, 0);
    cb(cur, "attn_out", 0);

    cur = build_lora_mm(layer.wo, cur);
    cb(cur, "wo_out", 0);

    cur = ggml_add(ctx0, cur, inpL);
    cb(cur, "attn_res", 0);

    inpL = cur;

    // Output projection
    {
        cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);

        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
    }
}

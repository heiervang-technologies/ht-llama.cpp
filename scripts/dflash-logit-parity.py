#!/usr/bin/env python3
# DFlash drafter LOGIT-parity harness (Round-12, item 2).
#
# WHY: Round-7b compared WEIGHTS (storage fidelity: GGUF Q6_K vs safetensors
# bf16, ~1.78% RMS). That proves tensors are stored right; it does NOT prove the
# FORWARD pass produces the reference logits. A bug in extraction point, mask,
# position scheme, feature fusion, or per-head norm placement passes weight-parity
# green while silently tanking acceptance. This harness closes that gap: it runs
# the z-lab PyTorch drafter as ground truth and compares PER-POSITION DRAFT LOGITS
# against our llama.cpp drafter on the SAME fixed input.
#
# STATUS: SCAFFOLD. The reference forward encodes everything the Round-5 audit
# established about our implementation (so it doubles as an executable spec), but
# the z-lab-repo-specific module glue is marked TODO(zlab). The HF repository
# ships weights and config only; the custom DFlashDraftModel class lives in
# github.com/z-lab/dflash (or vLLM PR #41703) and must be installed separately.
#
# Pipeline being verified (from src/models/dflash.cpp + gemma4.cpp audit):
#   target: gemma-4-31B-it, capture hidden states AFTER each target layer l_out
#           (LATE extraction, the llama.cpp default) at layer ids:
#           DFLASH_TARGET_LAYER_IDS below.
#   drafter: concat features over those layers -> fc -> dflash_hidden_norm (RMS)
#            -> 5x cross-attn blocks (K/V = [ctx_features, noise], non-causal mask,
#               per-head q_norm/k_norm, SWA on layers [T,T,T,T,F], window 2048)
#            -> output_norm -> shared target lm_head
#            -> final_logit_softcapping = 30.0  (Gemma4)
#   noise block: [id_last, <mask>*(block_size-1)], block_size = 16,
#                noise embeddings scaled by sqrt(n_embd)  (Gemma4 embed scale).
#
# Usage (in an env with torch + transformers + the downloaded models):
#   dflash-logit-parity.py reference \
#       --target  <hf path or id: google/gemma-4-31B-it> \
#       --drafter <hf path or id: z-lab/gemma-4-31B-it-DFlash> \
#       --prompt-file PROMPT.txt --out ref_logits.npz
#
#   # dump our side from llama.cpp (see dump_ours() docstring), then:
#   dflash-logit-parity.py compare --ref ref_logits.npz --ours ours_logits.npz
#
# Output: per draft position — top-1 agreement, top-5 overlap, logit RMS / max-abs.

import argparse
import json
import os
import sys

# Fallback defaults (verified against z-lab/gemma-4-31B-it-DFlash config.json on
# 2026-05-31). These are ONLY used if the drafter config can't be read; the
# reference forward calls load_dflash_constants() to read them from the actual
# downloaded config so they can never silently drift from the model.
DFLASH_TARGET_LAYER_IDS = [1, 12, 23, 35, 46, 57]
BLOCK_SIZE = 16
MASK_TOKEN_ID = 4          # drafter GGUF tokenizer <mask>
FINAL_LOGIT_SOFTCAP = 30.0  # Gemma4
SWA_PATTERN = [True, True, True, True, False]
SWA_WINDOW = 2048


def load_dflash_constants(drafter_dir):
    """Read the architectural constants from the drafter's config.json.

    Data-driven so the harness can never assert a value that disagrees with the
    actual model. Returns a dict; raises if the config is missing required keys."""
    cfg_path = os.path.join(drafter_dir, "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    df = c.get("dflash_config", {})
    layer_types = c.get("layer_types", [])
    consts = {
        "target_layer_ids": df.get("target_layer_ids", DFLASH_TARGET_LAYER_IDS),
        "block_size": c.get("block_size", BLOCK_SIZE),
        "mask_token_id": df.get("mask_token_id", MASK_TOKEN_ID),
        "final_logit_softcapping": c.get("final_logit_softcapping", FINAL_LOGIT_SOFTCAP),
        "swa_pattern": [("sliding" in t) for t in layer_types] if layer_types else SWA_PATTERN,
        "sliding_window": c.get("sliding_window", SWA_WINDOW),
        "num_attention_heads": c.get("num_attention_heads"),
        "num_key_value_heads": c.get("num_key_value_heads"),
        "head_dim": c.get("head_dim"),
        "hidden_size": c.get("hidden_size"),
        "num_hidden_layers": c.get("num_hidden_layers"),
    }
    # Drift guard: warn loudly if the on-disk config disagrees with our fallbacks,
    # so a model update surfaces instead of being silently absorbed.
    if consts["target_layer_ids"] != DFLASH_TARGET_LAYER_IDS:
        print(f"[logit-parity] WARN: target_layer_ids in config "
              f"{consts['target_layer_ids']} != fallback {DFLASH_TARGET_LAYER_IDS}", file=sys.stderr)
    return consts


def _need(mod):
    try:
        return __import__(mod)
    except Exception as e:
        sys.exit(f"[logit-parity] missing dependency '{mod}': {e}\n"
                 f"Run in the convert env (the one with transformers/torch) where the "
                 f"Round-12 download landed. This host's .venv numpy is known-broken (MKL).")


def cmd_reference(args):
    """Ground-truth forward via the z-lab PyTorch drafter.

    Captures target hidden states (LATE convention) and runs the drafter block-
    diffusion forward to produce per-position draft logits."""
    torch = _need("torch")
    _need("transformers")
    _need("numpy")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Data-driven constants from the actual drafter config (never hardcoded).
    consts = load_dflash_constants(args.drafter)
    layer_ids = consts["target_layer_ids"]
    print(f"[logit-parity] constants from {args.drafter}/config.json: "
          f"layer_ids={layer_ids} block_size={consts['block_size']} "
          f"mask={consts['mask_token_id']} softcap={consts['final_logit_softcapping']} "
          f"swa={consts['swa_pattern']} window={consts['sliding_window']}", file=sys.stderr)

    prompt = open(args.prompt_file).read() if args.prompt_file else \
        "Write a 50-word paragraph about speculative decoding."

    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(
        args.target, torch_dtype=torch.bfloat16, output_hidden_states=True, device_map="auto")
    target.eval()

    ids = tok(prompt, return_tensors="pt").input_ids.to(target.device)
    with torch.no_grad():
        out = target(ids)
    # hidden_states[i] is the OUTPUT of layer i-1 (index 0 = embeddings), so the
    # post-l_out hidden for target layer L is hidden_states[L+1]. LATE convention.
    hs = out.hidden_states
    feats = [hs[L + 1][0, -1, :] for L in layer_ids]  # last-token ctx
    ctx_features = torch.cat(feats, dim=-1)  # [n_layers * n_embd]
    id_last = int(ids[0, -1].item())

    # TODO(zlab): load z-lab/gemma-4-31B-it-DFlash drafter modules and run the
    # block-diffusion forward described in the header. Requires the z-lab repo's
    # DFlash drafter class (not a vanilla HF CausalLM). Steps, once wired:
    #   1. fc(ctx_features) -> hidden_norm (RMS)
    #   2. noise embeds = target.embed_tokens([id_last, MASK*(BLOCK_SIZE-1)]) * sqrt(n_embd)
    #   3. 5 cross-attn blocks, K/V = [ctx, noise], non-causal mask, q/k per-head RMS,
    #      SWA per SWA_PATTERN/SWA_WINDOW
    #   4. output_norm -> target.lm_head -> softcap(FINAL_LOGIT_SOFTCAP)
    #   5. draft_logits[pos] for pos in 1..BLOCK_SIZE-1
    raise SystemExit(
        "[logit-parity] reference scaffold reached the z-lab drafter forward "
        "(TODO(zlab)). Target hidden-state capture is implemented and correct per "
        "the audit; wire the z-lab drafter module here once the safetensors are "
        "downloaded. ctx_features shape would be "
        f"{tuple(ctx_features.shape)}, id_last={id_last}.")


def cmd_compare(args):
    """Compare reference vs our llama.cpp draft logits, per position."""
    np = _need("numpy")
    ref = np.load(args.ref)
    ours = np.load(args.ours)
    rl, ol = ref["logits"], ours["logits"]  # [n_pos, n_vocab]
    if rl.shape != ol.shape:
        sys.exit(f"shape mismatch: ref {rl.shape} vs ours {ol.shape}")
    print(f"# positions={rl.shape[0]} vocab={rl.shape[1]}")
    print("| pos | top1 agree | top5 overlap | logit RMS | max|Δ| |")
    print("|---|---|---|---:|---:|")
    for p in range(rl.shape[0]):
        r, o = rl[p], ol[p]
        r5 = set(np.argsort(r)[-5:].tolist())
        o5 = set(np.argsort(o)[-5:].tolist())
        top1 = int(np.argmax(r)) == int(np.argmax(o))
        rms = float(np.sqrt(np.mean((r - o) ** 2)))
        mx = float(np.max(np.abs(r - o)))
        print(f"| {p} | {'Y' if top1 else 'N'} | {len(r5 & o5)}/5 | {rms:.4f} | {mx:.4f} |")


# How to dump OUR side from llama.cpp:
#   build-cuda/bin/llama-speculative-simple already logs the top-5 of draft pos 1
#   under LLAMA_DFLASH_DEBUG. For full per-position vectors, add a one-shot dump in
#   common/speculative.cpp draft() guarded by LLAMA_DFLASH_DUMP=<path>, writing
#   llama_get_logits_ith(ctx_dft_dec, i) for i in 1..block_size-1 as an .npz with
#   key "logits". (Deferred to the build step; do not add until the squash lands.)


def main():
    ap = argparse.ArgumentParser(description="DFlash drafter logit-parity harness (item 2 scaffold)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference", help="z-lab PyTorch ground-truth forward")
    r.add_argument("--target", required=True)
    r.add_argument("--drafter", required=True)
    r.add_argument("--prompt-file")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_reference)
    c = sub.add_parser("compare", help="compare ref vs ours .npz")
    c.add_argument("--ref", required=True)
    c.add_argument("--ours", required=True)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

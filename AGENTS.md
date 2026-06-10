# Instructions for ht-llama.cpp

This is the [Heiervang Technologies](https://github.com/heiervang-technologies)
fork of [llama.cpp](https://github.com/ggml-org/llama.cpp). Unlike upstream,
**agentic contributions are welcome** — this fork judges code by quality, not
authorship. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution policy.

## Orient

- **Branch model**: `master` is a clean fast-forward of `upstream/master` —
  never commit directly. `ht` is the default branch and where all HT-specific
  work goes. Feature branches sprout from `ht`, get squash-merged back via PR.
- **No upstream contributions**: this fork is strictly downstream. Don't draft
  commits with upstream framing or open PRs against `ggml-org/llama.cpp`.
- **HT-specific surfaces**:
  - `tools/server/webui/` — SvelteKit 5 frontend (rebranded UI, AI workspace,
    sandbox terminals, artifact gallery, /images, doc mode, AI patch editor)
  - `tools/server/webui-tauri/` — Tauri 2 desktop shell (Linux + Android APK)
  - `tools/termd/` — Rust sandbox sidecar (gVisor + Docker + iptables LAN-drop)
  - `ggml/src/ggml-turboq*` and `ggml/src/ggml-cuda/turboq.{cu,cuh}` —
    TurboQuant KV cache types and fused CUDA kernels
- **Out-of-tree assets**: GGUF model files live under `$GGUFS` / `$MODELS`;
  never hard-code paths or look inside `models/`.
- **README.md `HT Fork Changes`** — canonical feature inventory; consult it
  before assuming a behaviour is upstream stock.

## Verify before committing

| Surface | Command |
|---|---|
| Tauri shell | `cd tools/server/webui-tauri/src-tauri && cargo check && cargo clippy` |
| termd | `cd tools/termd && cargo check && cargo clippy` |
| C++ / ggml / server | follow upstream's [build.md](docs/build.md) |

The embedded `llama-server` web UI is the upstream default, downloaded from
the `llama-ui` HF bucket at build time via `LLAMA_USE_PREBUILT_UI=ON`
(default). The product-facing UI lives in the standalone
[heierchat](https://github.com/heiervang-technologies/heierchat) repo.

## Conventions

- Conventional Commits (`feat(scope):`, `fix(scope):`, `chore(scope):` …).
- One commit per logical change; squash fix-ups before merging.
- Secrets stay in env / GitHub Secrets — never commit to history.
- Destructive actions (force-push, branch deletion, history rewrite) require
  explicit go-ahead from the maintainer.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide.

## Resources

- [HT Discussions](https://github.com/orgs/heiervang-technologies/discussions)
  — questions, RFCs, design threads
- [Fork Management Guide](https://github.com/orgs/heiervang-technologies/discussions/3)
  — sync workflow, release flow
- [README.md](README.md) — full HT Fork Changes inventory
- [tools/termd/README.md](tools/termd/README.md) — sandbox sidecar HTTP/WS
  reference

## Model capability for AI maintainers

Acting as the standing AI maintainer for this fork — multi-day debug arcs
(e.g. dflash audit rounds), coordinating multi-agent deploys, judging when
to act vs surface to the human — requires a model in the **Claude Opus
capability tier** (or equivalent). Less capable models tend to drift from
the goal, re-introduce already-disproven hypotheses, miss subtle issues
(double-applied framework transforms, etc.), and under-surface destructive
operations that need human approval.

Smaller / cheaper models are fine for **scoped tasks** — one-shot
implementations from a clearly-specified design, mechanical edits, bench
harness execution, lint fixes — when handed off by an Opus-class
maintainer with concrete instructions. They are **not** appropriate for
the standing maintainer role itself.

If you are running as the AI maintainer here and you are not in this
tier, surface that and propose a handoff rather than driving cutting-edge
work autonomously.

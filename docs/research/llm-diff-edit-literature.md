# LLM-Generated Code Edits: Literature and Design Survey

A grounding report for designing a diff-edit tool in an LLM backend. Covers diff-format semantics, research on LLM patching, observed failure modes, alternative edit representations, and the tool-use vs in-prompt tradeoff.

## 1. Unified diff format — strict vs lenient parsing

The canonical unified diff is consumed by GNU `patch`, whose matching behaviour is far more permissive than most people assume.

- `patch` first tries line numbers from the hunk header, then scans forward and backward for a block of lines whose **context** matches. If exact context fails, it drops one leading and trailing context line at a time, up to the **fuzz factor** ([GNU Diffutils — Helping patch Find Inexact Matches](https://www.gnu.org/s/diffutils/manual/html_node/Inexact.html)).
- `--fuzz=N` / `-F N` caps how many context lines may be ignored. Default is 2; a fuzz of 3 or more effectively throws away all context for a standard 3-line-context diff — dangerous ([GNU Diffutils — patch Options](https://www.gnu.org/s/diffutils/manual/html_node/patch-Options.html)).
- `-l` / `--ignore-white-space` makes any run of whitespace match any other run; non-whitespace still must match exactly ([patch(1) man page](https://www.man7.org/linux/man-pages/man1/patch.1.html)).
- Context-free normal diffs cannot be fuzzed reliably because `patch` can't distinguish drift from an intended change.
- Rejected hunks go to `.rej` files; the user is expected to hand-merge.

`git apply` is stricter by default ("no-fuzz") but offers `--3way`: when the patch records blob IDs (indexlines) that are known locally, git builds a synthetic index and falls back to a 3-way merge, possibly leaving conflict markers ([git-apply docs](https://git-scm.com/docs/git-apply); [git-am docs](https://git-scm.com/docs/git-am)). The `am.threeWay` config promotes this to default for `git am`. This is the single most battle-tested "textual edit recovery" path available, and any LLM diff tool ought to at least consider it as a fallback when strict apply fails and the original blob is known.

## 2. Research on LLM patching

- **SWE-bench** (Jimenez et al., 2023, [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)) established the canonical benchmark: agent output is a unified diff applied with `git apply`. A non-trivial fraction of agent failures are *patch application failures* rather than wrong-logic failures.
- **SWE-agent** (Yang et al., NeurIPS 2024, [arXiv:2405.15793](https://arxiv.org/abs/2405.15793)) explicitly rejected raw unified diffs in favour of a bespoke **Agent-Computer Interface (ACI)** with scoped commands: `open`, `goto`, `edit <start>:<end>`, `scroll_down`, plus a syntactic validator that rejects malformed edits before they hit disk. The paper's thesis is that LLMs need agent-shaped interfaces, not developer-shaped ones.
- **Agentless** (Xia et al., 2024, [arXiv:2407.01489](https://arxiv.org/abs/2407.01489)) uses a plain search/replace diff format after hierarchical localization, samples 40 patches per issue, filters by syntax and regression tests, then re-ranks. 40.7% on SWE-bench Lite with Claude 3.5 Sonnet at very low cost — evidence that a dumb format plus sampling beats clever agent loops.
- **Moatless Tools** ([github.com/aorwall/moatless-tools](https://github.com/aorwall/moatless-tools)) foregoes diffs altogether: the agent calls `StringReplace`, `CreateFile`, `AppendString` tools; the resulting git diff is what's submitted. This is the same shape as Anthropic's `str_replace_based_edit_tool`.
- **RepairBench** (Silva & Monperrus, [arXiv:2409.18952](https://arxiv.org/abs/2409.18952)) and **PATCHEVAL** ([arXiv:2511.11019](https://arxiv.org/pdf/2511.11019)) standardize APR leaderboards across Defects4J, GitBug-Java, QuixBugs, and real CVEs.
- **Diff-XYZ** (Glukhov et al., NeurIPS 2025 Workshop, [arXiv:2510.12487](https://arxiv.org/abs/2510.12487)) isolates the format variable: 1,000 real commits evaluated on Apply / Anti-Apply / Generate. Key findings: **udiff formats win for Apply and Anti-Apply; search/replace wins for Diff Generation; smaller open models benefit from modified udiff variants; explicit format instructions matter more than format choice for weaker models.** Claude 4 Sonnet and GPT-4.1 are near-perfect on Apply.
- **Aider's unified-diff post** ([aider.chat/docs/unified-diffs.html](https://aider.chat/docs/unified-diffs.html)) is the most cited industry data point: a refactoring benchmark went from 20% (SEARCH/REPLACE) to 61% (unified diffs without line numbers) on GPT-4 Turbo, primarily by reducing lazy "…" elisions 3x. The trick was dropping hunk line numbers entirely and demanding semantically coherent hunks.

## 3. Empirically observed failure modes

- **Line-number drift.** The Gemini CLI issue [#4836](https://github.com/google-gemini/gemini-cli/issues/4836) is a clean case study: patch headers' line counts disagree with the actual file even when the full file is in-context, and GPT-4o / o3 / Gemini-2.5 all fail to self-reconcile. Aider's response was to strip line numbers from hunk headers entirely ([aider.chat/docs/unified-diffs.html](https://aider.chat/docs/unified-diffs.html)).
- **Whitespace / tab mismatches.** GNU's answer (`patch -l`) exists precisely because mailers and editors mangle whitespace; LLMs do the same. The Anthropic text editor docs stress "unique matching" as a best practice ([docs.claude.com](https://docs.claude.com/en/docs/agents-and-tools/tool-use/text-editor-tool)).
- **Ambiguous matches.** When `old_string` appears multiple times, naive str_replace silently replaces the wrong one. Codex's `seek_sequence.rs` and Anthropic's tool docs both treat uniqueness as a precondition.
- **Lazy elisions** (`// ... rest unchanged`). Measured directly by Aider's 89-task laziness benchmark (20% → 61% by switching formats). Also documented in [Gemini CLI #4836](https://github.com/google-gemini/gemini-cli/issues/4836) as catastrophic — placeholder comments get written verbatim.
- **Invented context lines.** Diff-XYZ ([arXiv:2510.12487](https://arxiv.org/abs/2510.12487)) observes that smaller Qwen2.5-Coder models hallucinate context at the hunk boundaries even when old+diff are both fully specified.
- **Off-by-one indentation.** Catalogued under "syntax errors" in [arXiv:2406.08731](https://arxiv.org/html/2406.08731v1) and [arXiv:2407.06153](https://arxiv.org/html/2407.06153); especially harmful in Python where it flips semantics without triggering a parse error.

## 4. Alternatives to textual diffs

**AST-based edits** (tree-sitter). Parse the buffer, ask the model to emit a node-level operation (`replace_function "foo"`, `add_import "x"`), splice in the tree, re-serialize. Aider uses tree-sitter for repo maps and error-node linting ([aider.chat/2024/05/22/linting.html](https://aider.chat/2024/05/22/linting.html)); polyglot_ls ([github.com/PatWie/polyglot_ls](https://github.com/PatWie/polyglot_ls)) does full AST-aware prompts. Tradeoffs: precise, inherently whitespace-safe, syntactically valid by construction — but needs a parser per language, breaks on comments/macros, and AST node addressing is itself a format the model must learn.

**Line-anchor edits.** `replace_lines_N_to_M` style. Compact, simple to apply, but fragile: the model must know current line numbers, so any previous edit in the same turn shifts the anchor. SWE-agent mitigates this by making the editor re-print the window after every edit.

**Structured JSON operations.** `{op: "insert_after_line", line: 42, text: "..."}`. Robust to parsing (no ambiguous whitespace) and trivially validated, but verbose — a 10-line change can 3x the output token count versus a compact diff.

**Speculative "apply model"** (Cursor). The main model emits a *sloppy* edit (often with elisions); a small fast model rewrites the whole file conditioned on the sloppy edit, using speculative decoding where the original file is the draft (Fireworks blog, [fireworks.ai/blog/cursor](https://fireworks.ai/blog/cursor); foundations in [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)). Reported 1000+ tokens/s, 13x speedup over Llama-3-70b vanilla. Tradeoff: you now run two models and own the apply model's failure modes, but elisions become a feature rather than a bug.

**OpenAI V4A format** ([OpenAI Cookbook — GPT-4.1 Prompting Guide](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide), [Apply Patch docs](https://developers.openai.com/api/docs/guides/tools-apply-patch)) is a no-line-numbers diff variant with explicit `*** Begin Patch`/`*** Update File:` headers. Context-free-grammar parseable, trained into GPT-4.1/5/5.1.

## 5. Tool-use vs in-prompt edit formats

**Emit-as-text** (unified diff in an assistant message):
- Pros: cheapest tokens (3 lines of context vs a full JSON tool call); natural for models pretrained on git logs; model can batch many hunks in one turn.
- Cons: host must parse and recover from malformed output; no schema validation at decode time; concurrency with other tool calls is awkward.

**Structured tool call** (`str_replace`, `insert`, `create`):
- Pros: JSON-schema validation at decode time (or via grammar-constrained decoding — relevant for llama.cpp / llguidance); each edit is an atomic, individually-confirmable action; plays well with agent loops where every step expects tool feedback.
- Cons: higher per-edit overhead; chatty for multi-hunk refactors; models untrained on the specific schema can emit plausible-looking but wrong arguments (e.g. Diff-XYZ shows GPT-4.1 drifts back to V4A unless instructed).

Diff-XYZ's scaling result — bigger models prefer udiff, smaller open models prefer search/replace — maps directly onto this choice. A strong hosted model is cheapest in text-diff mode; a small local model (the relevant case for an llama.cpp backend) is more reliable with short, structured str_replace calls backed by grammar constraints.

## Design takeaways

1. **Offer two formats, not one.** A strict `str_replace`-style tool for small local models (backed by grammar constraints), and a unified-diff channel for stronger models. Let the caller pick; default based on model size.
2. **Drop hunk line numbers.** Aider and OpenAI V4A both do. Line numbers drift, context doesn't.
3. **Enforce uniqueness** of `old_string` at the apply layer, and return a structured error listing match count and nearby lines so the model can retry with more context.
4. **Fuzz like GNU patch, fall back like git.** On strict-apply failure: try whitespace-insensitive match (`-l` semantics), then context-trim fuzz up to F=2, then, if the original blob hash is known, a 3-way merge. Never silently exceed fuzz=2.
5. **Detect and reject elisions.** Regex for common placeholders (`\.\.\. rest`, `existing code`, `unchanged`) in any hunk's `+` lines and fail loudly — the Gemini CLI bug shows silent acceptance is catastrophic.
6. **Validate post-apply with tree-sitter.** Cheap, language-agnostic, catches off-by-one indentation and truncation. Feed ERROR nodes back to the model verbatim (Aider's linting loop).
7. **Consider an "apply model" path.** For a local-inference product, a small fine-tuned apply model running via speculative decoding against the original file is a natural fit for llama.cpp's speculative-decoding infrastructure.
8. **Log failure modes.** Line drift, ambiguous match, whitespace mismatch, elision, invented context, off-by-one — these six buckets cover nearly every real failure we found and make downstream evaluation tractable.

## Sources

- [GNU Diffutils — Helping patch Find Inexact Matches](https://www.gnu.org/s/diffutils/manual/html_node/Inexact.html)
- [GNU Diffutils — patch Options](https://www.gnu.org/s/diffutils/manual/html_node/patch-Options.html)
- [patch(1) man page](https://www.man7.org/linux/man-pages/man1/patch.1.html)
- [git-apply documentation](https://git-scm.com/docs/git-apply)
- [git-am documentation](https://git-scm.com/docs/git-am)
- [SWE-bench — arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- [SWE-agent — arXiv:2405.15793](https://arxiv.org/abs/2405.15793)
- [Agentless — arXiv:2407.01489](https://arxiv.org/abs/2407.01489)
- [Moatless Tools](https://github.com/aorwall/moatless-tools)
- [RepairBench — arXiv:2409.18952](https://arxiv.org/abs/2409.18952)
- [PATCHEVAL — arXiv:2511.11019](https://arxiv.org/pdf/2511.11019)
- [Diff-XYZ — arXiv:2510.12487](https://arxiv.org/abs/2510.12487)
- [Aider — Unified diffs make GPT-4 Turbo 3x less lazy](https://aider.chat/docs/unified-diffs.html)
- [Aider — Linting code for LLMs with tree-sitter](https://aider.chat/2024/05/22/linting.html)
- [Anthropic text editor tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/text-editor-tool)
- [OpenAI GPT-4.1 Prompting Guide (V4A format)](https://developers.openai.com/cookbook/examples/gpt4-1_prompting_guide)
- [OpenAI apply_patch tool](https://developers.openai.com/api/docs/guides/tools-apply-patch)
- [Fireworks — How Cursor built Fast Apply](https://fireworks.ai/blog/cursor)
- [Fast Inference from Transformers via Speculative Decoding — arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- [Gemini CLI issue #4836 — lazy elision case study](https://github.com/google-gemini/gemini-cli/issues/4836)
- [Where Do LLMs Fail When Generating Code? — arXiv:2406.08731](https://arxiv.org/html/2406.08731v1)
- [What is Wrong with Your Code Generated by LLMs? — arXiv:2407.06153](https://arxiv.org/html/2407.06153)

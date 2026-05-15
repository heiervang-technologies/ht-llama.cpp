/**
 * Per-turn primers for the front-end "Deep research" composer toggle.
 *
 * The mode does not introduce a new endpoint, tool, or backend code path —
 * it primes the existing agentic loop with a research-shaped system prompt
 * and bumps the max-turn ceiling so the loop has room to plan, search, and
 * synthesise instead of bailing after the default ~10 turns. All tool
 * invocations (web_search, fetch_url, fetch_image, MCP-provided tools)
 * remain the model's choice from the same registry the normal chat uses.
 */

/** Hard ceiling on per-turn research budget. We clamp below this whenever
 *  the active model's context window can't actually sustain it (see
 *  `computeDeepResearchTurnBudget`). 30 is generous; the loop terminates
 *  the moment the model emits a final answer with no tool calls. */
export const DEEP_RESEARCH_MAX_TURNS = 30;

/** Lower bound — even a tiny-context model gets a few turns so the user
 *  sees the mode behave differently from a normal chat. */
export const DEEP_RESEARCH_MIN_TURNS = 5;

/** Rough per-turn context cost: a tool call's prompt context grows by the
 *  tool result, which for web searches and fetched pages routinely lands
 *  in the 2-6K range. 4K is a conservative midpoint that errs toward
 *  fitting rather than overflowing. */
const TOKENS_PER_RESEARCH_TURN = 4000;

/** Reserved budget for the final synthesis turn (citations, structure,
 *  limitations section). We don't want the loop to fill ctx and leave no
 *  room to write the actual report. */
const RESERVED_TOKENS_FOR_SYNTHESIS = 4000;

/**
 * Pick a turn budget for deep-research mode that fits within the model's
 * advertised context window. Returns `DEEP_RESEARCH_MAX_TURNS` when the
 * context size is unknown — the loop has its own per-iteration abort on
 * context-overflow errors, so an over-estimate is safer than refusing
 * to run.
 */
export function computeDeepResearchTurnBudget(contextSize: number | null | undefined): number {
	if (!contextSize || contextSize <= 0) return DEEP_RESEARCH_MAX_TURNS;
	const available = contextSize - RESERVED_TOKENS_FOR_SYNTHESIS;
	const fits = Math.floor(available / TOKENS_PER_RESEARCH_TURN);
	if (fits >= DEEP_RESEARCH_MAX_TURNS) return DEEP_RESEARCH_MAX_TURNS;
	if (fits <= DEEP_RESEARCH_MIN_TURNS) return DEEP_RESEARCH_MIN_TURNS;
	return fits;
}

/**
 * Compose the one-shot system message for a deep-research turn. The
 * methodology is fixed; the context-budget paragraph adapts to whatever
 * `n_ctx` the active model reports so the model is told how much rope
 * it actually has.
 */
export function buildDeepResearchSystemPrompt(contextSize: number | null | undefined): string {
	const budgetLine =
		contextSize && contextSize > 0
			? `\nContext budget: you have approximately ${contextSize.toLocaleString()} tokens of context to work with, of which ~${RESERVED_TOKENS_FOR_SYNTHESIS.toLocaleString()} should stay free for the final synthesis. Plan your searches accordingly — prefer focused queries over broad ones, and fetch full pages only when the snippet is genuinely insufficient.\n`
			: '';
	return `You are operating in deep-research mode. The user has asked for a thorough, evidence-backed answer rather than a quick reply.${budgetLine}
Methodology — follow this order:

1. **Plan.** Open with a brief plan: list the 3-6 sub-questions you intend to investigate. The plan stays visible to the user.
2. **Search.** Use the available web search and fetch tools to gather primary sources. Prefer original documents, papers, official docs, and reputable reporting over aggregators. Run multiple searches with refined queries when the first results are thin.
3. **Read deeply.** When a source looks promising, fetch the full page rather than relying on the snippet. Extract the specific claims relevant to each sub-question.
4. **Triangulate.** Cross-check important claims across at least two independent sources. Flag any disagreement explicitly rather than papering over it.
5. **Synthesise.** Once you have enough material, produce a structured markdown report. Use headings per sub-question, inline citations as numeric footnotes \`[1]\`, \`[2]\`, ..., and a final \`## Sources\` list pairing each footnote with the URL.
6. **Acknowledge gaps.** End with a short "Limitations" section listing what you couldn't verify, what your sources disagreed on, and what would need a human follow-up.

Discipline:

- Tool calls are the primary work. Don't pad with prose between calls — the user sees the agentic timeline and prefers terse running commentary.
- Don't stop after one or two searches if the question genuinely needs more. The turn budget is high in this mode; spend it.
- Do not fabricate citations. Every \`[n]\` must correspond to a URL you actually fetched.
- If a tool call fails, retry with a different query before giving up. Don't silently drop a sub-question.
- Watch your context budget. If you sense you're running out of room, stop searching and synthesise with what you have rather than truncating mid-report.
- The final synthesis is your assistant content; it should be the last thing emitted, after all tool work is done.`;
}

/**
 * @deprecated Prefer `buildDeepResearchSystemPrompt(contextSize)` so the
 * prompt reflects the active model's context window. Kept as a fallback
 * when no context info is available.
 */
export const DEEP_RESEARCH_SYSTEM_PROMPT = buildDeepResearchSystemPrompt(null);
